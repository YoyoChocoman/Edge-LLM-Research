import json
import argparse
import numpy as np
from collections import defaultdict

def analyze_anomaly(json_path, start_idx=100, end_idx=115):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        return

    # Calculate the P50 latency for each position to pinpoint exactly which chunk the spike occurred in.
    position_deltas = defaultdict(list)
    for run in runs:
        chunks = run.get("chunks", [])
        for i in range(1, len(chunks)):
            t_prev = chunks[i-1]["timestamp_abs"]
            t_curr = chunks[i]["timestamp_abs"]
            position_deltas[i].append((t_curr - t_prev) * 1000.0)

    # Identify the exact index where the maximum P50 value occurs within the specified range.
    max_p50 = 0
    spike_index = -1
    for i in range(start_idx, end_idx + 1):
        if i in position_deltas:
            p50 = np.percentile(position_deltas[i], 50)
            if p50 > max_p50:
                max_p50 = p50
                spike_index = i

    print(f"\n[Anomaly Detection]")
    print(f"Targeting Window: Chunk {start_idx} ~ {end_idx}")
    print(f"Spike identified precisely at Chunk: {spike_index} (P50 Latency: {max_p50:.2f} ms)\n")

    print(f"=== Content generated at Chunk {spike_index} across all runs ===")
    print(f"{'Run':<5} | {'Latency(ms)':<12} | {'Content at Spike':<20} | {'Context (Prev 3 -> Spike -> Next 3)'}")
    print("-" * 90)

    # Print out exactly what text was output by each run at that peak point.
    for run in runs:
        chunks = run.get("chunks", [])
        if len(chunks) > spike_index + 3:
            delta_ms = (chunks[spike_index]["timestamp_abs"] - chunks[spike_index-1]["timestamp_abs"]) * 1000
            spike_content = repr(chunks[spike_index]["content"])

            prev_content = "".join(c["content"] for c in chunks[spike_index-3:spike_index])
            next_content = "".join(c["content"] for c in chunks[spike_index+1:spike_index+4])
            context = f"{repr(prev_content)} -> {spike_content} -> {repr(next_content)}"

            run_id = run.get("run_id", "N/A")
            print(f"{run_id:<5} | {delta_ms:<12.2f} | {spike_content:<20} | {context}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Dive into Chunk Latency Anomalies")
    parser.add_argument("json_file", help="Path to the JSON results file")
    # can customize the query area
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=120)
    args = parser.parse_args()

    analyze_anomaly(args.json_file, args.start, args.end)