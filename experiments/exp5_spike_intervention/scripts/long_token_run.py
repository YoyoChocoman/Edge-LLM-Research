import time
import json
from datetime import datetime
import argparse
from llama_cpp import Llama

MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MAX_TOKENS = 2000
RUNS = 20

LONG_PROMPT = (
    "Perform an exhaustive, multi-dimensional linguistic, etymological, philosophical, and syntactic dissection of the target word.\n\n"
    "<target_word>\nmitigate\n</target_word>\n\n"
    "<target_definition>\nMake a situation less severe, serious, or painful.\n</target_definition>\n\n"
    "<user_sentence>\nThe government implemented new flood defenses to mitigate the damage caused by heavy rains.\n</user_sentence>\n\n"
    "You must structure your response into the following 10 mandatory sections, writing at least 150 words per section with absolute technical rigor and zero summarization:\n"
    "1. Etymological origins, PIE roots, and historical evolution through Latin and Old French.\n"
    "2. Morphological breakdown (prefix, root, suffix) and grammatical category transitions.\n"
    "3. Syntactic valency, subcategorization frame, and argument structure in the user sentence.\n"
    "4. Semantic feature analysis using componential analysis and semantic primes.\n"
    "5. Pragmatic and illocutionary force analysis within political/governmental discourse contexts.\n"
    "6. Corpus-based collocational profile, including typical left- and right-neighbors.\n"
    "7. Comparative synonymy analysis distinguishing 'mitigate' from 'alleviate', 'ameliorate', 'assuage', and 'palliate'.\n"
    "8. Detailed logical evaluation of whether the user sentence applies the word correctly, addressing edge cases.\n"
    "9. Generation of 5 alternative sentences across different registers (formal, legal, colloquial, literary, technical) with commentary.\n"
    "10. Final comprehensive synthesis and formal judgment summary.\n\n"
    "Begin your exhaustive analysis now. Do not truncate."
)

def run_chunk_profiling(llm, prompt_text, run_id):
    messages = [
        {"role": "system", "content": "You are a highly analytical GRE linguistic judge and linguistic professor."},
        {"role": "user", "content": prompt_text}
    ]

    response_stream = llm.create_chat_completion(
        messages=messages, temperature=0.0, max_tokens=MAX_TOKENS, stream=True
    )

    chunks_data = []
    chunk_idx = 0
    t_first_content = None

    for chunk in response_stream:
        choices = chunk.get("choices", [])
        if not choices: continue

        content = choices[0].get("delta", {}).get("content", "")
        if content:
            t_current = time.perf_counter()
            if t_first_content is None:
                t_first_content = t_current
                chunks_data.append({"chunk_idx": chunk_idx, "content": content, "timestamp_abs": t_current, "is_first": True})
            else:
                chunks_data.append({"chunk_idx": chunk_idx, "content": content, "timestamp_abs": t_current, "is_first": False})
            chunk_idx += 1

    return {"run_id": run_id, "chunks": chunks_data}

def main():
    parser = argparse.ArgumentParser(description="Long Token Profiling Inference")
    parser.add_argument("-o", "--output", help="Custom output JSON path", default=None)
    args = parser.parse_args()

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = args.output or f"results/long_tokenrun_{timestamp_str}.json"

    print(f"Initializing Model (max_tokens: {MAX_TOKENS})...")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2560, flash_attn=True, verbose=False)

    sys_msg = "You are a highly analytical GRE linguistic judge and linguistic professor."
    input_tokens = len(llm.tokenize(f"{sys_msg}\n{LONG_PROMPT}".encode('utf-8'), add_bos=True))
    print(f"Input Tokens Length: {input_tokens}")

    print("Warming up model...")
    _ = run_chunk_profiling(llm, LONG_PROMPT, 0)

    print(f"\nRunning Long-Token Benchmark (N={RUNS})...")
    runs = []
    for i in range(1, RUNS + 1):
        print(f"-> Running inference {i}/{RUNS}...", end="", flush=True)
        res = run_chunk_profiling(llm, LONG_PROMPT, i)
        print(f" Done. (Generated chunks: {len(res['chunks'])})")
        runs.append(res)

    experiment_data = {
        "metadata": {"experiment": "Long Token Profiling (>1024)", "input_tokens": input_tokens},
        "runs": runs
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    print(f"\nData saved to {results_file}")

if __name__ == "__main__":
    main()