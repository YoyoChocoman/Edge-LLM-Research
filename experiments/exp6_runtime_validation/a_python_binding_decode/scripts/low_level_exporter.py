import json
import time
from datetime import datetime
from llama_cpp import Llama

MODEL_PATH = "../../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
RESULTS_FILE = f"results/low_level_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"

WARMUP_TOKENS = 5
MAX_GENERATE = 500
RUNS = 20

EOS_TOKEN_IDS = {128001, 128009}

PROMPT = (
    "Evaluate if the user's sentence uses the target word correctly.\n\n"
    "<target_word>\nmitigate\n</target_word>\n\n"
    "<target_definition>\nMake a situation less severe\n</target_definition>\n\n"
    "<user_sentence>\nThe government implemented new flood defenses to "
    "mitigate the damage caused by heavy rains.\n</user_sentence>\n\n"
    "Execution Steps for 'reasoning':\n"
    "1. Check the Part of Speech.\n"
    "2. Check the Semantics.\n"
    "3. Make a final judgment.\n\n"
    "Provide an extremely detailed, step-by-step reasoning."
)


def token_text(llm: Llama, token_id: int) -> str:
    return llm.detokenize([token_id]).decode("utf-8", errors="replace")


def run_low_level_inference(
    llm: Llama,
    input_tokens: list[int],
    max_generate: int,
) -> dict:
    llm.reset()

    prefill_start_ns = time.perf_counter_ns()
    llm.eval(input_tokens)
    prefill_end_ns = time.perf_counter_ns()

    current_token = llm.sample(temp=0.0)
    step_data = []

    for step in range(1, max_generate + 1):
        eval_start_ns = time.perf_counter_ns()
        llm.eval([current_token])
        eval_end_ns = time.perf_counter_ns()

        sample_start_ns = time.perf_counter_ns()
        next_token = llm.sample(temp=0.0)
        sample_end_ns = time.perf_counter_ns()

        step_data.append(
            {
                "decode_step": step,
                # n_tokens is measured after current_token enters the KV state.
                "n_past_after_eval": llm.n_tokens,
                "evaluated_token_id": current_token,
                "evaluated_token_text": token_text(llm, current_token),
                "sampled_next_token_id": next_token,
                "sampled_next_token_text": token_text(llm, next_token),
                "eval_ms": (eval_end_ns - eval_start_ns) / 1_000_000,
                "sample_ms": (sample_end_ns - sample_start_ns) / 1_000_000,
            }
        )

        if next_token in EOS_TOKEN_IDS:
            break

        current_token = next_token

    return {
        "input_tokens": len(input_tokens),
        "prefill_ms": (prefill_end_ns - prefill_start_ns) / 1_000_000,
        "decode_steps": step_data,
    }


def main() -> None:
    print("Loading model for Exp6-A low-level binding measurement...")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    # Replace this with the exact rendered/tokenized prompt from Exp3–5
    # if direct cross-experiment comparison is required.
    input_tokens = llm.tokenize(PROMPT.encode("utf-8"), add_bos=True)

    print(f"Input tokens: {len(input_tokens)}")
    print(f"Warm-up: {WARMUP_TOKENS} generated tokens")
    run_low_level_inference(llm, input_tokens, WARMUP_TOKENS)

    runs_data = []
    for run_id in range(1, RUNS + 1):
        print(f"Run {run_id}/{RUNS}")
        result = run_low_level_inference(llm, input_tokens, MAX_GENERATE)
        result["run_id"] = run_id
        runs_data.append(result)

    experiment_data = {
        "metadata": {
            "experiment": "Exp6-A: Low-Level Python Binding Decode Measurement",
            "purpose": (
                "Tests whether the context-aligned latency spike persists "
                "after bypassing high-level completion/chat formatting and "
                "streaming iteration."
            ),
            "limitations": (
                "This still uses llama-cpp-python and does not by itself "
                "exclude Python-binding overhead or GPU synchronization effects."
            ),
            "model_path": MODEL_PATH,
            "sampling": "greedy, temperature=0.0",
            "num_runs": RUNS,
            "max_generate": MAX_GENERATE,
        },
        "results": {
            "Prompt_A_Control": {
                "input_tokens": len(input_tokens),
                "runs": runs_data,
            }
        },
    }

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2)

    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()