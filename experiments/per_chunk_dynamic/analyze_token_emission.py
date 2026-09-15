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
    # verbose=False 避免洗版
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

        # 使用 Llama 原生 Tokenizer 計算累積字串的真實 Token 數
        # add_bos=False 避免每次計算都重複加上 Beginning of Sentence token
        current_tokens = len(llm.tokenize(cumulative_string.encode('utf-8'), add_bos=False))

        # 計算邊際 Token 增加量
        delta_tokens = current_tokens - prev_tokens

        # 計算延遲 (跳過第一個 chunk，因其無前驅時間)
        latency_ms = 0.0
        if i > 0:
            latency_ms = (chunk["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0

        # 只印出觀察區間內的結果
        if start_idx <= i <= end_idx:
            # 將換行符號替換為可視字元，避免破壞表格排版
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