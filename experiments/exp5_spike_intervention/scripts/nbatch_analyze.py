import json
import argparse
import numpy as np
from collections import defaultdict

def detect_all_spikes(positions, p50_array, threshold_z=3.0, min_val=9.0):
    mean_val = np.mean(p50_array)
    std_val = np.std(p50_array)
    threshold = max(mean_val + threshold_z * std_val, min_val)

    spikes = []
    for i in range(1, len(p50_array) - 1):
        if p50_array[i] > threshold and p50_array[i] > p50_array[i-1] and p50_array[i] > p50_array[i+1]:
            spikes.append((positions[i], p50_array[i]))
    return spikes

def analyze_nbatch_results(json_path, window_size=20):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data.get("results", {})
    prompt_groups = defaultdict(dict)

    for key, item in results.items():
        n_batch = item["n_batch"]
        prompt_name = item["prompt_name"]
        input_tokens = item.get("input_tokens", 0)
        runs = item.get("runs", [])

        position_deltas = defaultdict(list)
        contents = {}

        for run in runs:
            chunks = run.get("chunks", [])
            for i in range(1, len(chunks)):
                delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0
                position_deltas[i].append(delta_ms)
                if run["run_id"] == 1:
                    contents[i] = chunks[i].get("content", "")

        valid_positions = sorted([p for p in position_deltas.keys() if p > 30])
        if not valid_positions:
            prompt_groups[prompt_name][n_batch] = {
                "input_tokens": input_tokens,
                "spikes": [],
                "position_deltas": position_deltas,
                "contents": contents
            }
            continue

        p50_array = np.array([np.percentile(position_deltas[p], 50) for p in valid_positions])
        spikes = detect_all_spikes(valid_positions, p50_array)

        prompt_groups[prompt_name][n_batch] = {
            "input_tokens": input_tokens,
            "spikes": spikes,
            "position_deltas": position_deltas,
            "contents": contents
        }

    for prompt_name, n_batch_dict in sorted(prompt_groups.items()):
        print("=" * 110)
        print(f"PROMPT: {prompt_name}")
        print("=" * 110)

        print("[Part 1: Spike Locations Comparison across n_batch]")
        print(f"{'Experiment':<18} | {'In Toks':<7} | {'Spike Chunk':<11} | {'Total Ctx':<9} | {'Spike P50':<10} | {'Content'}")
        print("-" * 100)

        for n_batch, p_data in sorted(n_batch_dict.items(), key=lambda x: str(x[0])):
            exp_name = f"n_batch: {n_batch}"
            toks = p_data.get("input_tokens", 0)
            spikes = p_data.get("spikes", [])
            contents = p_data.get("contents", {})

            if not spikes:
                print(f"{exp_name:<18} | {toks:<7} | {'No spikes detected':<51}")
            else:
                for idx, (spike_pos, spike_p50) in enumerate(spikes):
                    total_context = toks + spike_pos
                    disp_content = repr(contents.get(spike_pos, ""))[:15]
                    f_label = exp_name if idx == 0 else ""
                    t_label = str(toks) if idx == 0 else ""
                    print(f"{f_label:<18} | {t_label:<7} | {spike_pos:<11} | {total_context:<9} | {spike_p50:<8.2f}ms | {disp_content}")
            print("-" * 100)

        print("\n[Part 2: P50 Step-Up Analysis Around Spikes]")

        for n_batch, p_data in sorted(n_batch_dict.items(), key=lambda x: str(x[0])):
            exp_name = f"n_batch: {n_batch}"
            toks = p_data.get("input_tokens", 0)
            spikes = p_data.get("spikes", [])
            deltas = p_data.get("position_deltas", {})

            print(f"  -> {exp_name} (Input Tokens: {toks})")
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
                    if i in deltas: before_vals.extend(deltas[i])
                for i in range(after_start, after_end + 1):
                    if i in deltas: after_vals.extend(deltas[i])

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
    parser = argparse.ArgumentParser(description="n_batch Comprehensive Spike & Step-Up Analysis")
    parser.add_argument("json_file", help="Path to the n_batch JSON results file")
    args = parser.parse_args()

    analyze_nbatch_results(args.json_file)