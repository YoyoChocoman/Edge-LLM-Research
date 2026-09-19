import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_latency(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_tokens = data.get("metadata", {}).get("input_tokens", 0)
    runs = data.get("runs", [])

    position_deltas = defaultdict(list)
    for run in runs:
        chunks = run.get("chunks", [])
        for i in range(1, len(chunks)):
            delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
            position_deltas[i].append(delta_ms)

    valid_positions = sorted([p for p in position_deltas.keys() if p > 10])
    if not valid_positions:
        print("No data available to plot.")
        return

    total_contexts = [input_tokens + p for p in valid_positions]
    p50_vals = [np.percentile(position_deltas[p], 50) for p in valid_positions]
    p90_vals = [np.percentile(position_deltas[p], 90) for p in valid_positions]
    p95_vals = [np.percentile(position_deltas[p], 95) for p in valid_positions]

    plt.figure(figsize=(12, 6))
    plt.plot(total_contexts, p50_vals, color='blue', label='P50 Latency')
    plt.plot(total_contexts, p90_vals, color='orange', linestyle='--', label='P90 Latency')
    plt.plot(total_contexts, p95_vals, color='red', linestyle=':', label='P95 Latency')

    plt.axvline(x=input_tokens, color='gray', linestyle='-.', alpha=0.7, label=f'Input Ends ({input_tokens} toks)')

    plt.title("Long-Token Streaming Latency Dynamics", fontsize=14)
    plt.xlabel("Total Context Length (Input Tokens + Output Chunk Index)", fontsize=12)
    plt.ylabel("Inter-Chunk Latency $\Delta t_i$ (ms)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_file = "figures/long_token_latency_plot.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Latency plot successfully saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot long token profiling latency dynamics")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()

    plot_latency(args.json_file)