import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_json_file(json_path):
    """解析新版多 Prompt JSON 格式檔案"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    fa_bool = metadata.get("flash_attention", None)

    if fa_bool is True:
        exp_name = "Flash Attention: ON"
    elif fa_bool is False:
        exp_name = "Flash Attention: OFF"
    else:
        exp_name = metadata.get("experiment", os.path.basename(json_path))

    results = data.get("results", {})
    return exp_name, results

def plot_aligned_comparison(file1, file2):
    """讀取兩個檔案，並繪製 6x2（或 2x6）的對齊比較圖，以 Total Context 進行對齊"""
    name1, results1 = parse_json_file(file1)
    name2, results2 = parse_json_file(file2)

    all_prompts = sorted(list(set(results1.keys()).union(set(results2.keys()))))
    num_prompts = len(all_prompts)

    if num_prompts == 0:
        print("No prompts found in the JSON files.")
        return

    fig, axes = plt.subplots(num_prompts, 2, figsize=(16, 4 * num_prompts), sharex=False, sharey=True)
    fig.suptitle("Per-Chunk Streaming Latency Dynamics Aligned by Total Context Length", fontsize=16, y=0.98)

    if num_prompts == 1:
        axes = np.array([axes])

    datasets = [
        (name1, results1, axes[:, 0]),
        (name2, results2, axes[:, 1])
    ]

    for exp_name, results_dict, col_axes in datasets:
        for ax, prompt_name in zip(col_axes, all_prompts):
            prompt_data = results_dict.get(prompt_name, {})
            input_tokens = prompt_data.get("input_tokens", 0)
            runs = prompt_data.get("runs", [])

            position_deltas = defaultdict(list)
            for run in runs:
                chunks = run.get("chunks", [])
                for i in range(1, len(chunks)):
                    total_context = input_tokens + i
                    delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                    position_deltas[total_context].append(delta_ms)

            if not position_deltas:
                ax.set_title(f"{prompt_name} ({exp_name}) - No Data")
                continue

            positions = sorted(position_deltas.keys())
            p50_vals = [np.percentile(position_deltas[p], 50) for p in positions]
            p90_vals = [np.percentile(position_deltas[p], 90) for p in positions]
            p95_vals = [np.percentile(position_deltas[p], 95) for p in positions]

            ax.plot(positions, p50_vals, color='blue', label='P50')
            ax.plot(positions, p90_vals, color='orange', linestyle='--', label='P90')
            ax.plot(positions, p95_vals, color='red', linestyle=':', label='P95')

            ax.axvline(x=input_tokens, color='gray', linestyle='-.', alpha=0.6, label='Input Ends')

            ax.set_title(f"[{prompt_name}] {exp_name}\n(Input Tokens: {input_tokens})", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Total Context Length (Input Tokens + Output Chunk Index)")

    fig.text(0.02, 0.5, "Inter-Chunk Latency $Delta t_i$ (ms)", va='center', rotation='vertical', fontsize=12)

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    output_path = "figures/aligned_comparison_6x2.png"
    plt.savefig(output_path, dpi=300)
    print(f"Aligned comparison plot successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Aligned Latency Dynamics Comparison (6x2)")
    parser.add_argument("file1", help="Path to the first JSON results file (e.g., FA-ON)")
    parser.add_argument("file2", help="Path to the second JSON results file (e.g., FA-OFF)")
    args = parser.parse_args()

    plot_aligned_comparison(args.file1, args.file2)