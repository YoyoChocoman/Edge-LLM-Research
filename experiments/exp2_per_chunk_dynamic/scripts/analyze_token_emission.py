import json
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from llama_cpp import Llama

def analyze_token_emission(json_path, model_path, run_idx=0, start_idx=100, end_idx=120):
    print(f"Loading experiment data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs or len(runs) <= run_idx:
        print("Invalid run index or no runs found.")
        return

    target_run = runs[run_idx]
    chunks = target_run.get("chunks", [])

    print(f"Loading Model for exact Tokenization: {model_path}")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    print(f"\nAnalyzing Run ID {target_run.get('run_id')} (Window: Chunk {start_idx} to {end_idx})")
    print("-" * 85)
    print(f"{'Chunk':<7} | {'Latency(ms)':<12} | {'Cum. Tokens':<12} | {'Δ Tokens':<10} | {'Content'}")
    print("-" * 85)

    cumulative_string = ""
    prev_tokens = 0

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        cumulative_string += content

        # Calculate the actual number of tokens for an accumulated string using the native Llama tokenizer
        # add_bos=False avoids repeatedly adding the beginning of sentence token during every calculation.
        current_tokens = len(llm.tokenize(cumulative_string.encode('utf-8'), add_bos=False))

        # calculate delta token
        delta_tokens = current_tokens - prev_tokens

        latency_ms = 0.0
        if i > 0:
            latency_ms = (chunk["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0

        if start_idx <= i <= end_idx:
            display_content = repr(content)
            print(f"{i:<7} | {latency_ms:<12.2f} | {current_tokens:<12} | {delta_tokens:<10} | {display_content}")

        prev_tokens = current_tokens

    print("-" * 85)
    print("Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Token Emission per Chunk")
    parser.add_argument("json_file", help="Path to the JSON results file")
    parser.add_argument("--model", type=str, default="../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=115)
    args = parser.parse_args()

    analyze_token_emission(args.json_file, args.model, start_idx=args.start, end_idx=args.end)