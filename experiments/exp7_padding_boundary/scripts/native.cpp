#include "llama.h"
#ifdef EXP7_TRACE
#include "exp7-trace.h"
#endif
#include <chrono>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}

void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

void fill(llama_batch & b, const std::vector<llama_token> & ts, int pos) {
    b.n_tokens = static_cast<int>(ts.size());
    for (int i=0; i<b.n_tokens; ++i) {
        b.token[i ] = ts[i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0; b.logits[i] = (i == b.n_tokens-1);
    }
}

llama_token sample(llama_context * ctx) {
    auto * s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    check(s != nullptr, "sampler allocation failed");
    llama_sampler_chain_add(s, llama_sampler_init_penalties(64, 1.0f, 0.0f, 0.0f));
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    const auto t = llama_sampler_sample(s, ctx, -1);
    llama_sampler_free(s);
    return t;
}

struct Row {
    int run, step, before, after, token, expected, sampled;
    double remove_ms, setup_ms, decode_ms, eval_ms, sync_ms, sample_ms, core_ms;
};


constexpr int WARMUP = 5;
const std::string MODEL = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
const std::string SEQUENCE = "results/sequence.txt";

int main(int argc, char ** argv) {
    // Command-line configuration
    check(argc == 5, "Usage: native async native.json events.json runs");
    const int runs = std::stoi(argv[4]);
    check(runs > 0, "runs must be positive");

    const std::string mode = argv[1];
    check(mode == "async", "Exp7 uses async mode");

    const std::string output = argv[2];

    // Fixed experiment checks
    std::ifstream in(SEQUENCE);
    int np = 0, ns = 0;
    check(bool(in >> np >> ns), "Invalid sequence header");
    check(np > 0 && np <= 512 && ns > 0 && np + ns <= 2048, "Sequence exceeds context or prefill batch");

    std::vector<llama_token> prompt(np);
    std::vector<llama_token> tokens(ns);
    std::vector<llama_token> expected(ns);

    for (auto & t : prompt) check(bool(in >> t),"Truncated prompt");
    for (int i = 0; i < ns; ++i) check(bool(in >> tokens[i] >> expected[i]), "Truncated decode sequence");

    // llama.cpp initialization
    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = INT_MAX;

    auto * model = llama_model_load_from_file(MODEL.c_str(), mp);
    check(model != nullptr, "Model load failed");

    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = true;
    cp.no_perf = false;

    auto * ctx = llama_init_from_model(model, cp);
    check(ctx != nullptr, "Context creation failed");

    llama_log_set(
        [](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::cerr << text;
        },
        nullptr
    );

    // Runtime state
    auto mem = llama_get_memory(ctx);
    auto batch = llama_batch_init(512, 0, 1);
    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(runs) * ns);
#ifdef EXP7_TRACE
    exp7_trace_reset(static_cast<size_t>(runs) * ns * 16);
#endif

    // Warmup + measured runs
    for (int run = 0; run <= runs; ++run) {
        if (run == 0 && WARMUP == 0) continue;

        llama_synchronize(ctx);
        check(llama_memory_seq_rm(mem, -1, 0, -1), "Reset failed");

        // Prefill prompt.
        fill(batch, prompt, 0);
        check(llama_decode(ctx, batch) == 0, "Prefill failed");

        const int count = (run == 0) ? WARMUP : ns;

        for (int j = 0; j < count; ++j) {
            const int before = np + j;
            check(llama_memory_seq_pos_max(mem, 0) == before - 1, "Unexpected KV position before decode");

#ifdef EXP7_TRACE
            if (run) exp7_trace_begin(run, j + 1, before);
#endif
            // Sequence removal
            const auto a = Clock::now();
            check(llama_memory_seq_rm(mem, -1, before, -1), "Sequence removal failed");

            // Batch setup
            const auto b = Clock::now();
            fill(batch, {tokens[j]}, before);

            // Native llama.cpp decode
            const auto c = Clock::now();
            check(llama_decode(ctx, batch) == 0, "Decode failed");

            // Optional synchronization
            const auto d =Clock::now();

            // No explicit synchronization in the Exp7 async comparison.

            // Sampling
            const auto e = Clock::now();
            const auto sampled =sample(ctx);

            // Verify native state
            const auto f = Clock::now();
#ifdef EXP7_TRACE
            if (run) exp7_trace_end();
#endif
            const int after = llama_memory_seq_pos_max(mem, 0) + 1;
            check(after == before + 1, "Unexpected KV position after decode");

            // Store measured runs
            if (run) {
                rows.push_back({
                    run,
                    j + 1,
                    before,
                    after,
                    tokens[j],
                    expected[j],
                    sampled,
                    ms(a, b),
                    ms(b, c),
                    ms(c, d),
                    ms(a, d),
                    ms(d, e),
                    ms(e, f),
                    ms(a, f)
                });
            }
        }
    }

#ifdef EXP7_TRACE
    check(exp7_trace_write(argv[3]), "Cannot write events JSON");
#endif
    // Write timing JSON
    std::ofstream out(output);

    out << "{\n";
    out << "  \"experiment\": \"Exp7: Effective padding boundary intervention\",\n";
    out << "  \"padding\": " << EXP7_PADDING_SIZE << ",\n";
    out << "  \"warmup_steps\": " << WARMUP << ",\n";
    out << "  \"context\": {\"n_ctx\": " << cp.n_ctx
        << ", \"n_batch\": " << cp.n_batch << ", \"n_ubatch\": " << cp.n_ubatch
        << ", \"flash_attn_type\": " << static_cast<int>(cp.flash_attn_type)
        << ", \"offload_kqv\": " << cp.offload_kqv << "},\n";
    out << "  \"prompt_tokens\": [";
    for (int i = 0; i < np; ++i) out << (i ? ", " : "") << prompt[i];
    out << "],\n";
    out << "  \"mode\": \"" << mode << "\",\n";
    out << "  \"input_tokens\": " << np << ",\n";
    out << "  \"decode_steps\": " << ns << ",\n";
    out << "  \"runs\": [\n";

    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & r = rows[i];

        out << "    {\n";
        out << "      \"run_id\": " << r.run << ",\n";
        out << "      \"decode_step\": " << r.step << ",\n";
        out << "      \"n_past_before_eval\": " << r.before << ",\n";
        out << "      \"n_past_after_eval\": " << r.after << ",\n";
        out << "      \"evaluated_token_id\": " << r.token << ",\n";
        out << "      \"expected_next_token_id\": " << r.expected << ",\n";
        out << "      \"sampled_next_token_id\": " << r.sampled << ",\n";
        out << "      \"remove_ms\": " << r.remove_ms << ",\n";
        out << "      \"setup_ms\": " << r.setup_ms << ",\n";
        out << "      \"decode_ms\": " << r.decode_ms << ",\n";
        out << "      \"eval_ms\": " << r.eval_ms << ",\n";
        out << "      \"sync_ms\": " << r.sync_ms << ",\n";
        out << "      \"sample_ms\": " << r.sample_ms << ",\n";
        out << "      \"core_decode_ms\": " << r.core_ms << "\n";
        out << "    }";

        if (i + 1 < rows.size()) out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    out.close();
    check(bool(out), "Cannot write timing JSON");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}