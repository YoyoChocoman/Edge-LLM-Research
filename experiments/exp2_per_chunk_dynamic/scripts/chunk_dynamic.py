import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from llama_cpp import Llama

MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MAX_TOKENS = 500
RUNS = 50
RESULTS_FILE = f"results/exp_chunk_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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

def run_chunk_profiling(llm, run_id):
    messages = [
        {"role": "system", "content": "You are a highly analytical GRE linguistic judge."},
        {"role": "user", "content": PROMPT}
    ]

    t_llm_start = time.perf_counter()

    response_stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        stream=True
    )

    chunks_data = []
    chunk_idx = 0
    t_first_content = None
    full_content = ""

    for chunk in response_stream:
        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            t_current = time.perf_counter()
            full_content += content

            if t_first_content is None:
                t_first_content = t_current
                # TTFT is calculated independently, and inter-chunk deltas are not calculated for this chunk.
                chunks_data.append({
                    "chunk_idx": chunk_idx,
                    "content": content,
                    "content_length": len(content),
                    "timestamp_abs": t_current,
                    "is_first": True
                })
            else:
                chunks_data.append({
                    "chunk_idx": chunk_idx,
                    "content": content,
                    "content_length": len(content),
                    "timestamp_abs": t_current,
                    "is_first": False
                })
            chunk_idx += 1

    # Derived aggregate metrics
    ttft_ms = (t_first_content - t_llm_start) * 1000 if t_first_content else None
    t_last_content = chunks_data[-1]["timestamp_abs"] if chunks_data else None
    decode_ms = (t_last_content - t_first_content) * 1000 if (t_last_content and t_first_content) else None

    true_output_tokens = len(llm.tokenize(full_content.encode('utf-8'))) if full_content else 0

    return {
        "run_id": run_id,
        "aggregate_metrics": {
            "ttft_ms": ttft_ms,
            "decode_duration_ms": decode_ms,
            "chunk_count": chunk_idx,
            "true_output_tokens": true_output_tokens
        },
        "chunks": chunks_data
    }

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    print(f"Loading Model: {MODEL_PATH}")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    print("Warming up... (Cold start isolation)")
    _ = run_chunk_profiling(llm, run_id=0)

    experiment_data = {
        "metadata": {
            "experiment": "Per-Chunk Streaming Latency Dynamics (Pilot)",
            "model": "Llama-3-8B-Instruct-Q4_K_M",
            "max_tokens_setting": MAX_TOKENS,
            "runs": RUNS,
            "methodology_note": "Timestamps are absolute (perf_counter) upon Python receiving the non-empty chunk."
        },
        "runs": []
    }

    print(f"\nStarting Pilot Benchmark (N={RUNS})...")

    for run in range(1, RUNS + 1):
        print(f"Executing Run {run}/{RUNS}...", end=" ", flush=True)
        res = run_chunk_profiling(llm, run_id=run)
        experiment_data["runs"].append(res)

        agg = res["aggregate_metrics"]
        print(f"Done. TTFT: {agg['ttft_ms']:.2f}ms | Chunks: {agg['chunk_count']} | Tokens: {agg['true_output_tokens']}")

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)

    print(f"\nPilot complete. Raw data saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()