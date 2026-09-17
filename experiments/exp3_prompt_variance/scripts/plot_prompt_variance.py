import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_dynamics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results:
        return

    # build 1x3 plot
    fig, axes = plt.subplots(1, len(results), figsize=(18, 6), sharey=True)
    fig.suptitle("Per-Chunk Streaming Latency Dynamics by Prompt", fontsize=14)

    for ax, (prompt_name, prompt_data) in zip(axes, results.items()):
        runs = prompt_data.get("runs", [])
        position_deltas = defaultdict(list)

        scatter_x = []
        scatter_y = []

        for run in runs:
            chunks = run.get("chunks", [])
            for i in range(1, len(chunks)):
                delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                position_deltas[i].append(delta_ms)
                scatter_x.append(i)
                scatter_y.append(delta_ms)

        positions = sorted(position_deltas.keys())
        p50_vals = [np.percentile(position_deltas[p], 50) for p in positions]
        p90_vals = [np.percentile(position_deltas[p], 90) for p in positions]
        p95_vals = [np.percentile(position_deltas[p], 95) for p in positions]

        ax.scatter(scatter_x, scatter_y, alpha=0.1, color='gray', s=10)
        ax.plot(positions, p50_vals, color='blue', label='P50')
        ax.plot(positions, p90_vals, color='orange', linestyle='--', label='P90')
        ax.plot(positions, p95_vals, color='red', linestyle=':', label='P95')

        ax.set_title(f"{prompt_name}\n(Input Tokens: {prompt_data.get('input_tokens')})")
        ax.set_xlabel("Generation Position (i)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[0].set_ylabel("Inter-Chunk Latency $Delta t_i$ (ms)")
    plt.tight_layout()
    plt.savefig("figures/prompt_variance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot P50/P90/P95 for Exp 3")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()
    plot_dynamics(args.json_file)