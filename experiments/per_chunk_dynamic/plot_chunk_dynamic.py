# scripts/profiling/plot_chunk_dynamics.py
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_chunk_dynamics(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    position_deltas = defaultdict(list)

    runs = data.get("runs", [])
    if not runs:
        print("No runs found in data.")
        return

    # 1. elta T
    for run in runs:
        chunks = run.get("chunks", [])
        for i in range(1, len(chunks)):
            t_prev = chunks[i-1]["timestamp_abs"]
            t_curr = chunks[i]["timestamp_abs"]
            delta_ms = (t_curr - t_prev) * 1000.0

            position_deltas[i].append(delta_ms)

    # 2. P50, P90, P95
    positions = sorted(position_deltas.keys())
    p50_values = []
    p90_values = []
    p95_values = []

    # Collect all x (position) and y (delta_ms) values ​​to plot a scatter chart.
    scatter_x = []
    scatter_y = []

    for pos in positions:
        deltas = position_deltas[pos]

        scatter_x.extend([pos] * len(deltas))
        scatter_y.extend(deltas)

        p50_values.append(np.percentile(deltas, 50))
        p90_values.append(np.percentile(deltas, 90))
        p95_values.append(np.percentile(deltas, 95))

    # 3. Visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(scatter_x, scatter_y, alpha=0.15, color='gray', label='Raw $\Delta t_i$ Observations', s=15)
    plt.plot(positions, p50_values, color='blue', linewidth=2, label='P50 (Median)')
    plt.plot(positions, p90_values, color='orange', linewidth=1.5, linestyle='--', label='P90')
    plt.plot(positions, p95_values, color='red', linewidth=1.5, linestyle=':', label='P95 (Tail Latency)')

    plt.title(f"Per-Chunk Streaming Latency Dynamics (N={len(runs)})")
    plt.xlabel("Generation Position ($i$)")
    plt.ylabel("Inter-Chunk Latency $\Delta t_i$ (ms)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    y_max = max(p95_values) * 1.5
    plt.ylim(0, y_max)

    plt.tight_layout()
    plt.savefig("figure/tpot_residuals.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Per-Chunk Latency Dynamics")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()

    analyze_chunk_dynamics(args.json_file)