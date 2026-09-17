import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_aligned_dynamics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results: return

    # build 3x1 plot，with the same X and Y axis
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 10), sharex=True, sharey=True)
    fig.suptitle("Latency Dynamics Aligned by Total Context Length", fontsize=14)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for ax, (prompt_name, prompt_data) in zip(axes, results.items()):
        input_tokens = prompt_data.get("input_tokens", 0)
        runs = prompt_data.get("runs", [])

        position_deltas = defaultdict(list)
        for run in runs:
            chunks = run.get("chunks", [])
            for i in range(1, len(chunks)):
                total_context = input_tokens + i
                delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                position_deltas[total_context].append(delta_ms)

        positions = sorted(position_deltas.keys())
        p50_vals = [np.percentile(position_deltas[p], 50) for p in positions]
        p90_vals = [np.percentile(position_deltas[p], 90) for p in positions]
        p95_vals = [np.percentile(position_deltas[p], 95) for p in positions]

        ax.plot(positions, p50_vals, color='blue', label='P50')
        ax.plot(positions, p90_vals, color='orange', linestyle='--', label='P90')
        ax.plot(positions, p95_vals, color='red', linestyle=':', label='P95')

        ax.axvline(x=input_tokens, color='gray', linestyle='-.', alpha=0.5, label='Input Ends')

        ax.set_title(f"{prompt_name} (Input: {input_tokens} toks)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Total Context Length (Input Tokens + Output Chunk Index)")
    fig.text(0.04, 0.5, "Inter-Chunk Latency $Delta t_i$ (ms)", va='center', rotation='vertical')

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
    plt.savefig("figures/aligned.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Aligned Latency Dynamics")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()
    plot_aligned_dynamics(args.json_file)