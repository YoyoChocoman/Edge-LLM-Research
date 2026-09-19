import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_nbatch_grid(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    prompt_groups = defaultdict(dict)

    for key, item in results.items():
        n_batch = item.get("n_batch")
        prompt_name = item.get("prompt_name")
        prompt_groups[prompt_name][n_batch] = item

    all_prompts = sorted(list(prompt_groups.keys()))
    all_nbatches = sorted(list(set(item.get("n_batch") for item in results.values())))

    num_prompts = len(all_prompts)
    num_batches = len(all_nbatches)

    if num_prompts == 0 or num_batches == 0:
        print("No valid data found to plot.")
        return

    fig, axes = plt.subplots(num_prompts, num_batches, figsize=(5 * num_batches, 4 * num_prompts), sharex=False, sharey=True)
    fig.suptitle("Per-Chunk Streaming Latency Dynamics by n_batch Settings", fontsize=16, y=0.98)

    if num_prompts == 1 and num_batches == 1:
        axes = np.array([[axes]])
    elif num_prompts == 1:
        axes = np.array([axes])
    elif num_batches == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, prompt_name in enumerate(all_prompts):
        for col_idx, n_batch_val in enumerate(all_nbatches):
            ax = axes[row_idx, col_idx]
            item = prompt_groups[prompt_name].get(n_batch_val, {})
            input_tokens = item.get("input_tokens", 0)
            runs = item.get("runs", [])

            position_deltas = defaultdict(list)
            for run in runs:
                chunks = run.get("chunks", [])
                for i in range(1, len(chunks)):
                    total_context = input_tokens + i
                    delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                    position_deltas[total_context].append(delta_ms)

            if not position_deltas:
                ax.set_title(f"[{prompt_name}] n_batch: {n_batch_val}\nNo Data", fontsize=10)
                continue

            positions = sorted(position_deltas.keys())
            p50_vals = [np.percentile(position_deltas[p], 50) for p in positions]
            p90_vals = [np.percentile(position_deltas[p], 90) for p in positions]
            p95_vals = [np.percentile(position_deltas[p], 95) for p in positions]

            ax.plot(positions, p50_vals, color='blue', label='P50')
            ax.plot(positions, p90_vals, color='orange', linestyle='--', label='P90')
            ax.plot(positions, p95_vals, color='red', linestyle=':', label='P95')

            ax.axvline(x=input_tokens, color='gray', linestyle='-.', alpha=0.6, label='Input Ends')

            ax.set_title(f"[{prompt_name}] n_batch: {n_batch_val}\n(Input Tokens: {input_tokens})", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Total Context Length (Input Tokens + Output Chunk Index)")

    fig.text(0.01, 0.5, "Inter-Chunk Latency $\Delta t_i$ (ms)", va='center', rotation='vertical', fontsize=12)

    output_path = "figures/nbatch_grid_comparison.png"
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"n_batch grid comparison plot successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="n_batch Grid Latency Plotter")
    parser.add_argument("json_file", help="Path to the n_batch JSON results file")
    args = parser.parse_args()

    plot_nbatch_grid(args.json_file)