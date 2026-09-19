import json
import argparse
import numpy as np
from collections import defaultdict

def detect_all_spikes(positions, p50_array, threshold_z=2.5, min_val=8.5):
    mean_val = np.mean(p50_array)
    std_val = np.std(p50_array)
    threshold = max(mean_val + threshold_z * std_val, min_val)

    spikes = []
    for i in range(1, len(p50_array) - 1):
        if p50_array[i] > threshold and p50_array[i] > p50_array[i-1] and p50_array[i] > p50_array[i+1]:
            spikes.append((positions[i], p50_array[i]))
    return spikes

def analyze_spikes(json_path, window_size=20):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_tokens = data.get("metadata", {}).get("input_tokens", 0)
    runs = data.get("runs", [])

    position_deltas = defaultdict(list)
    contents = {}

    for run in runs:
        chunks = run.get("chunks", [])
        for i in range(1, len(chunks)):
            delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
            position_deltas[i].append(delta_ms)
            if run["run_id"] == 1:
                contents[i] = chunks[i]["content"]

    valid_positions = sorted([p for p in position_deltas.keys() if p > 10])
    if not valid_positions:
        print("No valid positions for spike analysis.")
        return

    p50_array = np.array([np.percentile(position_deltas[p], 50) for p in valid_positions])
    spikes = detect_all_spikes(valid_positions, p50_array)

    # Part 1: Spike Locations & Overview
    print("=" * 95)
    print(f"[Part 1: Spike Locations Comparison] (Total Output Chunks: {len(valid_positions)})")
    print("=" * 95)
    print(f"{'Chunk Index':<12} | {'Total Context':<14} | {'Spike P50':<12} | {'Content Preview'}")
    print("-" * 95)

    if not spikes:
        print("No significant spikes detected with current threshold.")
    else:
        for chunk_pos, p50_val in spikes:
            total_ctx = input_tokens + chunk_pos
            preview = repr(contents.get(chunk_pos, ""))[:25]
            print(f"{chunk_pos:<12} | {total_ctx:<14} | {p50_val:<10.2f}ms | {preview}")
    print("=" * 95)

    # Part 2: P50 Step-Up Analysis Around Spikes
    print("\n" + "=" * 95)
    print("[Part 2: P50 Step-Up Analysis Around Spikes]")
    print("=" * 95)

    if not spikes:
        print("No significant spikes detected to perform step-up analysis.")
        return

    print(f"{'Spike Chunk':<12} | {'Total Context':<14} | {'Before P50':<12} | {'After P50':<12} | {'Delta (ms)'}")
    print("-" * 95)

    for spike_pos, _ in spikes:
        total_ctx = input_tokens + spike_pos
        before_start, before_end = spike_pos - window_size - 5, spike_pos - 5
        after_start, after_end = spike_pos + 5, spike_pos + window_size + 5

        before_vals = []
        after_vals = []

        for i in range(before_start, before_end + 1):
            if i in position_deltas: before_vals.extend(position_deltas[i])
        for i in range(after_start, after_end + 1):
            if i in position_deltas: after_vals.extend(position_deltas[i])

        if before_vals and after_vals:
            p50_before = np.percentile(before_vals, 50)
            p50_after = np.percentile(after_vals, 50)
            diff = p50_after - p50_before
            print(f"{spike_pos:<12} | {total_ctx:<14} | {p50_before:<12.4f} | {p50_after:<12.4f} | {diff:+.4f} ms")
        else:
            print(f"{spike_pos:<12} | {total_ctx:<14} | {'N/A':<12} | {'N/A':<12} | {'N/A'}")
    print("-" * 95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze long token profiling spikes & step-up dynamics")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()

    analyze_spikes(args.json_file)