import json
import argparse
import numpy as np

def verify_step_up(json_path, spike_idx=108, window_size=50):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    runs = data.get("runs", [])
    if not runs:
        return

    # define a clean observation window to avoid spike's potential volatility preceding and following it
    before_start, before_end = spike_idx - window_size - 5, spike_idx - 5
    after_start, after_end = spike_idx + 5, spike_idx + window_size + 5

    all_before = []
    all_after = []
    run_stats = []

    for run in runs:
        chunks = run.get("chunks", [])
        before_run = []
        after_run = []

        for i in range(1, len(chunks)):
            delta_ms = (chunks[i]["timestamp_abs"] - chunks[i-1]["timestamp_abs"]) * 1000.0

            if before_start <= i <= before_end:
                before_run.append(delta_ms)
            elif after_start <= i <= after_end:
                after_run.append(delta_ms)

        all_before.extend(before_run)
        all_after.extend(after_run)

        if before_run and after_run:
            p50_before = np.percentile(before_run, 50)
            p50_after = np.percentile(after_run, 50)
            diff = p50_after - p50_before
            run_stats.append({
                "run_id": run.get("run_id"),
                "p50_before": p50_before,
                "p50_after": p50_after,
                "diff": diff
            })

    # Aggregate Analysis
    print(f"\n=== Aggregate Statistical Analysis ===")
    print(f"Before Spike Window (Chunks {before_start} ~ {before_end})")
    print(f"After Spike Window  (Chunks {after_start} ~ {after_end})")
    print("-" * 50)

    global_p50_before = np.percentile(all_before, 50)
    global_p50_after = np.percentile(all_after, 50)

    print(f"Global P50 Before  : {global_p50_before:.4f} ms")
    print(f"Global P50 After   : {global_p50_after:.4f} ms")
    print(f"Step-up Delta      : {global_p50_after - global_p50_before:+.4f} ms")
    print(f"Step-up percentage : {(global_p50_after - global_p50_before) / global_p50_before * 100:+.2f} %")

    # Per-Run Consistency
    print(f"\n=== Per-Run Consistency Check ===")
    print(f"{'Run':<5} | {'Before P50(ms)':<15} | {'After P50(ms)':<15} | {'Delta(ms)':<10}")
    print("-" * 55)

    increased_count = 0
    for stat in run_stats:
        print(f"{stat['run_id']:<5} | {stat['p50_before']:<15.4f} | {stat['p50_after']:<15.4f} | {stat['diff']:+7.4f}")
        if stat['diff'] > 0:
            increased_count += 1

    total_runs = len(run_stats)
    consistency_rate = (increased_count / total_runs) * 100
    print("-" * 55)
    print(f"Runs demonstrating a Step-Up: {increased_count} / {total_runs} ({consistency_rate:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Step-up Latency After Anomaly")
    parser.add_argument("json_file", help="Path to the JSON results file")
    args = parser.parse_args()

    verify_step_up(args.json_file)