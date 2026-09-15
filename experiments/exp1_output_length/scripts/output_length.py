import os
import time
import json
from datetime import datetime

from llama_cpp import Llama

MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

OUTPUT_LIMITS = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]
RUNS_PER_SETTING = 20
RESULTS_FILE = f"results/exp_output_length_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Fixed Prompt
PROMPT = (
    "Evaluate if the user's sentence uses the target word correctly.\n\n"
    "<target_word>\nmitigate\n</target_word>\n\n"
    "<target_definition>\nMake a situation less severe\n</target_definition>\n\n"
    "<user_sentence>\nThe government implemented new flood defenses to mitigate the damage caused by heavy rains.\n</user_sentence>\n\n"
    "Execution Steps for 'reasoning':\n"
    "1. Check the Part of Speech.\n"
    "2. Check the Semantics.\n"
    "3. Make a final judgment.\n\n"
    "Provide an extremely detailed, step-by-step reasoning."
)

def run_single_inference(llm, max_tokens):
    messages = [
        {"role": "system", "content": "You are a highly analytical GRE linguistic judge."},
        {"role": "user", "content": PROMPT}
    ]

    t_llm_start = time.perf_counter()

    response_stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        stream=True
    )

    result_str = ""
    chunk_count = 0
    t_first_content = None
    t_last_content = None

    for chunk in response_stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            current_time = time.perf_counter()
            if t_first_content is None:
                t_first_content = current_time
            t_last_content = current_time

            result_str += content
            chunk_count += 1

    # Derived Metrics
    actual_tokens = len(llm.tokenize(result_str.encode('utf-8'))) if result_str else 0

    ttft_ms = (t_first_content - t_llm_start) * 1000 if t_first_content else 0.0
    decode_ms = (t_last_content - t_first_content) * 1000 if (t_last_content and t_first_content) else 0.0
    tpot_ms = decode_ms / (actual_tokens - 1) if actual_tokens > 1 else None

    return {
        "max_tokens_setting": max_tokens,
        "actual_tokens": actual_tokens,
        "chunk_count": chunk_count,
        "ttft_ms": round(ttft_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "tpot_ms": round(tpot_ms, 2) if tpot_ms else None,
        "is_truncated": actual_tokens >= max_tokens
    }

def main():
    print(f"Loading Engine: {MODEL_PATH}")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    print("Warming up... (Cold start isolation)")
    _ = run_single_inference(llm, max_tokens=10)

    experiment_data = {
        "metadata": {
            "experiment": "LLM Output Length Characterization",
            "model": "Llama-3-8B-Instruct-Q4_K_M",
            "prompt_length_approx": len(PROMPT.split())
        },
        "raw_runs": []
    }

    print("\nStarting Benchmark...")
    print(f"{'MaxLimit':<10} | {'ActualTok':<10} | {'TTFT(ms)':<10} | {'Decode(ms)':<12} | {'TPOT(ms)':<10}")
    print("-" * 65)

    for limit in OUTPUT_LIMITS:
        for run in range(RUNS_PER_SETTING):
            res = run_single_inference(llm, max_tokens=limit)
            experiment_data["raw_runs"].append({
                "run_idx": run + 1,
                **res
            })

            tpot_str = f"{res['tpot_ms']:.2f}" if res['tpot_ms'] else "N/A"
            print(f"{limit:<10} | {res['actual_tokens']:<10} | {res['ttft_ms']:<10.2f} | {res['decode_ms']:<12.2f} | {tpot_str:<10}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\nExperiment complete. Data saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()