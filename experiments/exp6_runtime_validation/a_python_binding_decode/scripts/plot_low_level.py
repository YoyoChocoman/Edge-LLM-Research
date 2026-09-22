import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_dynamics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        print("No results found in JSON.")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6), sharey=True)
    if len(results) == 1:
        axes = [axes]

    fig.suptitle("Per-Step Evaluation Latency Dynamics and Percentiles", fontsize=14, fontweight='bold')

    for ax, (prompt_name, prompt_data) in zip(axes, results.items()):
        runs = prompt_data.get("runs", [])
        position_deltas = defaultdict(list)

        scatter_x = []
        scatter_y = []

        for run in runs:
            steps = run.get("decode_steps", [])
            for step in steps:
                pos = step["decode_step"]
                eval_ms = step["eval_ms"]  # 抓取底層推論耗時
                position_deltas[pos].append(eval_ms)
                scatter_x.append(pos)
                scatter_y.append(eval_ms)

        positions = sorted(position_deltas.keys())
        p50_vals = [np.percentile(position_deltas[p], 50) for p in positions]
        p90_vals = [np.percentile(position_deltas[p], 90) for p in positions]
        p95_vals = [np.percentile(position_deltas[p], 95) for p in positions]

        # 繪圖標準：灰色散佈 + P50藍線 + P90橘虛線 + P95紅點線
        ax.scatter(scatter_x, scatter_y, alpha=0.15, color='gray', s=10, label='Raw Samples')
        ax.plot(positions, p50_vals, color='blue', linewidth=2, label='P50')
        ax.plot(positions, p90_vals, color='orange', linestyle='--', linewidth=2, label='P90')
        ax.plot(positions, p95_vals, color='red', linestyle=':', linewidth=2, label='P95')

        ax.set_title(f"{prompt_name}\n(Input Tokens: {prompt_data.get('input_tokens')})")
        ax.set_xlabel("Generation Position (Decode Step)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[0].set_ylabel("Step Evaluation Latency $\Delta t_i$ (ms)")

    os.makedirs("figures", exist_ok=True)
    output_path = "figures/low_level_variance.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot P50/P90/P95 for new Low-Level Benchmark format.")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()
    plot_dynamics(args.json_file)