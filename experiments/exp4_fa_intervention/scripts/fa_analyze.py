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

def parse_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    fa_bool = metadata.get("flash_attention", None)

    if fa_bool is True:
        exp_name = "flash_attention: on"
    elif fa_bool is False:
        exp_name = "flash_attention: off"

    results = data.get("results", {})
    parsed_data = {}

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

        valid_positions = sorted([p for p in position_deltas.keys() if p > 30])
        if not valid_positions:
            parsed_data[prompt_name] = {
                "input_tokens": input_tokens,
                "spikes": [],
                "position_deltas": position_deltas,
                "contents": contents
            }
            continue

        p50_array = np.array([np.percentile(position_deltas[p], 50) for p in valid_positions])
        spikes = detect_all_spikes(valid_positions, p50_array)

        parsed_data[prompt_name] = {
            "input_tokens": input_tokens,
            "spikes": spikes,
            "position_deltas": position_deltas,
            "contents": contents
        }

    return exp_name, parsed_data

def analyze_and_compare(file1, file2, window_size=20):
    name1, data1 = parse_json_file(file1)
    name2, data2 = parse_json_file(file2)

    all_prompts = sorted(list(set(data1.keys()).union(set(data2.keys()))))

    for prompt_name in all_prompts:
        print("=" * 100)
        print(f"PROMPT: {prompt_name}")
        print("=" * 100)

        p_data1 = data1.get(prompt_name, {})
        p_data2 = data2.get(prompt_name, {})

        toks1 = p_data1.get("input_tokens", 0)
        spikes1 = p_data1.get("spikes", [])
        contents1 = p_data1.get("contents", {})
        deltas1 = p_data1.get("position_deltas", {})

        toks2 = p_data2.get("input_tokens", 0)
        spikes2 = p_data2.get("spikes", [])
        contents2 = p_data2.get("contents", {})
        deltas2 = p_data2.get("position_deltas", {})

        # Part 1: Spike Locations Comparison Table
        print("[Part 1: Spike Locations Comparison]")
        print(f"{'Experiment':<22} | {'In Toks':<7} | {'Spike Chunk':<11} | {'Total Ctx':<9} | {'Spike P50':<10} | {'Content'}")
        print("-" * 95)

        if not spikes1:
            print(f"{name1:<22} | {toks1:<7} | {'No spikes detected':<51}")
        else:
            for idx, (spike_pos, spike_p50) in enumerate(spikes1):
                total_context = toks1 + spike_pos
                disp_content = repr(contents1.get(spike_pos, ""))[:15]
                f_label = name1 if idx == 0 else ""
                t_label = str(toks1) if idx == 0 else ""
                print(f"{f_label:<22} | {t_label:<7} | {spike_pos:<11} | {total_context:<9} | {spike_p50:<8.2f}ms | {disp_content}")

        print("-" * 95)

        if not spikes2:
            print(f"{name2:<22} | {toks2:<7} | {'No spikes detected':<51}")
        else:
            for idx, (spike_pos, spike_p50) in enumerate(spikes2):
                total_context = toks2 + spike_pos
                disp_content = repr(contents2.get(spike_pos, ""))[:15]
                f_label = name2 if idx == 0 else ""
                t_label = str(toks2) if idx == 0 else ""
                print(f"{f_label:<22} | {t_label:<7} | {spike_pos:<11} | {total_context:<9} | {spike_p50:<8.2f}ms | {disp_content}")

        # Part 2: P50 Step-Up Analysis
        print("\n[Part 2: P50 Step-Up Analysis Around Spikes]")

        datasets = [
            (name1, toks1, spikes1, deltas1),
            (name2, toks2, spikes2, deltas2)
        ]

        for exp_name, input_tokens, spikes, position_deltas in datasets:
            print(f"  -> {exp_name} (Input Tokens: {input_tokens})")
            if not spikes:
                print("     No significant spikes detected.\n")
                continue

            print(f"     {'Spike Chunk':<12} | {'Before P50':<12} | {'After P50':<12} | {'Delta (ms)'}")
            print(f"     {'-' * 53}")

            for spike_pos, _ in spikes:
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
                    print(f"     {spike_pos:<12} | {p50_before:<12.4f} | {p50_after:<12.4f} | {diff:+.4f} ms")
                else:
                    print(f"     {spike_pos:<12} | {'N/A':<12} | {'N/A':<12} | {'N/A'}")
            print()
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Prompt Comprehensive Spike & Step-Up Analysis")
    parser.add_argument("file1", help="Path to the first JSON results file")
    parser.add_argument("file2", help="Path to the second JSON results file")
    args = parser.parse_args()

    analyze_and_compare(args.file1, args.file2)