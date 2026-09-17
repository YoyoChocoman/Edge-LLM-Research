import json
import argparse
import numpy as np
from collections import defaultdict

def get_spikes(valid_positions, p50_array, threshold_z=3.0, min_val=9.0):
    mean_val = np.mean(p50_array)
    std_val = np.std(p50_array)
    threshold = max(mean_val + threshold_z * std_val, min_val)
    spikes = []
    for i in range(1, len(p50_array) - 1):
        if p50_array[i] > threshold and p50_array[i] > p50_array[i-1] and p50_array[i] > p50_array[i+1]:
            spikes.append(valid_positions[i])
    return spikes

def analyze_step_ups(json_path, window_size=20):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results: return

    for prompt_name, prompt_data in results.items():
        print(f"\n{'='*70}")
        print(f"Prompt: {prompt_name} (Input Tokens: {prompt_data.get('input_tokens')})")
        print(f"{'='*70}")

        runs = prompt_data.get("runs", [])
        position_deltas = defaultdict(list)

        for run in runs:
            chunks = run.get("chunks", [])
            for i in range(1, len(chunks)):
                delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                position_deltas[i].append(delta_ms)

        valid_positions = sorted([p for p in position_deltas.keys() if p > 30])
        if not valid_positions: continue

        p50_array = np.array([np.percentile(position_deltas[p], 50) for p in valid_positions])
        spikes = get_spikes(valid_positions, p50_array)

        if not spikes:
            print("No significant spikes detected.")
            continue

        print(f"{'Spike Chunk':<12} | {'Before P50':<12} | {'After P50':<12} | {'Delta (ms)'}")
        print("-" * 55)

        for spike_idx in spikes:
            before_start, before_end = spike_idx - window_size - 5, spike_idx - 5
            after_start, after_end = spike_idx + 5, spike_idx + window_size + 5

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
                print(f"{spike_idx:<12} | {p50_before:<12.4f} | {p50_after:<12.4f} | {diff:+.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze P50 Step-up across Multiple Spikes")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()
    analyze_step_ups(args.json_file)