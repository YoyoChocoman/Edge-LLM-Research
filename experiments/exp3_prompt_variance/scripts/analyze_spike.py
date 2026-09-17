import json
import argparse
import numpy as np
from collections import defaultdict

def detect_all_spikes(positions, p50_array, threshold_z=3.0, min_val=9.0):
    """Identify all local maxima using Z-scores and absolute thresholds"""
    mean_val = np.mean(p50_array)
    std_val = np.std(p50_array)
    threshold = max(mean_val + threshold_z * std_val, min_val)

    spikes = []
    for i in range(1, len(p50_array) - 1):
        if p50_array[i] > threshold and p50_array[i] > p50_array[i-1] and p50_array[i] > p50_array[i+1]:
            spikes.append((positions[i], p50_array[i]))
    return spikes

def analyze_spikes(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    if not results: return

    print(f"{'Prompt Type':<20} | {'In Toks':<7} | {'Spike Chunk':<11} | {'Total Ctx':<9} | {'Spike P50':<10} | {'Content'}")
    print("-" * 90)

    for prompt_name, prompt_data in results.items():
        input_tokens = prompt_data.get("input_tokens", 0)
        runs = prompt_data.get("runs", [])

        position_deltas = defaultdict(list)
        contents = {}

        for run in runs:
            chunks = run.get("chunks", [])
            for i in range(1, len(chunks)):
                delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                position_deltas[i].append(delta_ms)
                if run["run_id"] == 1:
                    contents[i] = chunks[i]["content"]

        # ignore the first 30 warm-up chunk
        valid_positions = sorted([p for p in position_deltas.keys() if p > 30])
        if not valid_positions: continue

        p50_array = np.array([np.percentile(position_deltas[p], 50) for p in valid_positions])

        spikes = detect_all_spikes(valid_positions, p50_array)

        if not spikes:
            print(f"{prompt_name:<20} | {input_tokens:<7} | {'No spikes detected':<47}")
        else:
            for idx, (spike_pos, spike_p50) in enumerate(spikes):
                total_context = input_tokens + spike_pos
                disp_content = repr(contents.get(spike_pos, ""))[:15]
                p_label = prompt_name if idx == 0 else ""
                i_label = str(input_tokens) if idx == 0 else ""
                print(f"{p_label:<20} | {i_label:<7} | {spike_pos:<11} | {total_context:<9} | {spike_p50:<8.2f}ms | {disp_content}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Multiple Spike Locations")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()
    analyze_spikes(args.json_file)