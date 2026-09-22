import argparse
import json
import numpy as np

def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")

def local_reference(step_map, target_step, window):
    """Use a robust local median, excluding the target itself."""
    neighbors = []

    for offset in range(-window, window + 1):
        if offset == 0:
            continue

        neighbor_step = target_step + offset
        if neighbor_step in step_map:
            neighbors.append(step_map[neighbor_step])

    return float(np.median(neighbors)) if neighbors else None


def classify(spike_rate):
    if spike_rate >= 0.8:
        return "Consistent spike"
    if spike_rate >= 0.2:
        return "Partial / tail-latency event"
    return "Not consistently reproduced"


def analyze_results(json_path, target_steps, window, ratio_threshold, abs_threshold):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    metadata = data.get("metadata", {})
    results = data.get("results", {})

    print("=" * 100)
    print("LOW-LEVEL LOCAL SPIKE ANALYSIS")
    print(f"Experiment: {metadata.get('experiment', 'N/A')}")
    print(f"Targets: {target_steps} | Local window: ±{window} steps")
    print(
        f"Spike rule: target > local median × {ratio_threshold} "
        f"and excess > {abs_threshold} ms"
    )
    print("=" * 100)

    for prompt_name, prompt_data in results.items():
        runs = prompt_data.get("runs", [])

        print(f"\nPrompt: {prompt_name}")
        print(f"Input tokens: {prompt_data.get('input_tokens', 'N/A')}")
        print(f"Available runs: {len(runs)}")
        print("-" * 100)

        for target_step in target_steps:
            target_values = []
            reference_values = []
            excess_values = []
            ratios = []
            n_past_before_values = []
            n_past_after_values = []
            spike_count = 0
            valid_runs = 0

            for run in runs:
                steps = run.get("decode_steps", [])

                step_map = {
                    item["decode_step"]: item["eval_ms"]
                    for item in steps
                    if "decode_step" in item and "eval_ms" in item
                }

                step_metadata = {
                    item["decode_step"]: item
                    for item in steps
                    if "decode_step" in item
                }

                if target_step not in step_map:
                    continue

                reference = local_reference(step_map, target_step, window)
                if reference is None or reference <= 0:
                    continue

                valid_runs += 1
                target_value = step_map[target_step]
                item = step_metadata[target_step]

                n_past_after = item.get("n_past_after_eval")
                n_past_before = item.get(
                    "n_past_before_eval",
                    n_past_after - 1 if n_past_after is not None else None,
                )

                excess = target_value - reference
                ratio = target_value / reference
                is_spike = (
                    ratio > ratio_threshold
                    and excess > abs_threshold
                )

                target_values.append(target_value)
                reference_values.append(reference)
                excess_values.append(excess)
                ratios.append(ratio)

                if n_past_before is not None:
                    n_past_before_values.append(n_past_before)
                if n_past_after is not None:
                    n_past_after_values.append(n_past_after)

                if is_spike:
                    spike_count += 1

            if valid_runs == 0:
                print(f"Step {target_step}: no valid observations.")
                continue

            spike_rate = spike_count / valid_runs

            print(f"\nTarget decode step: {target_step}")
            print(
                "Internal position: "
                f"before={np.median(n_past_before_values) if n_past_before_values else 'N/A'}, "
                f"after={np.median(n_past_after_values) if n_past_after_values else 'N/A'}"
            )
            print(
                f"Target latency:   "
                f"P50={percentile(target_values, 50):.3f} ms | "
                f"P90={percentile(target_values, 90):.3f} ms | "
                f"P95={percentile(target_values, 95):.3f} ms"
            )
            print(
                f"Local reference:  "
                f"P50={percentile(reference_values, 50):.3f} ms"
            )
            print(
                f"Excess latency:   "
                f"median={np.median(excess_values):+.3f} ms | "
                f"P95={percentile(excess_values, 95):+.3f} ms"
            )
            print(
                f"Relative latency: "
                f"median={np.median(ratios):.2f}×"
            )
            print(
                f"Spike rate:       {spike_count}/{valid_runs} "
                f"({spike_rate:.1%}) → {classify(spike_rate)}"
            )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust local spike analysis for Exp6 low-level JSON results."
    )
    parser.add_argument("json_file", help="Path to experiment JSON file")
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[1, 134, 390],
        help="Decode steps to inspect",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Neighboring steps on each side for local reference",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.2,
        help="Required target/reference ratio to call a spike",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=0.8,
        help="Required absolute excess latency in milliseconds",
    )

    args = parser.parse_args()

    analyze_results(
        json_path=args.json_file,
        target_steps=args.targets,
        window=args.window,
        ratio_threshold=args.ratio_threshold,
        abs_threshold=args.abs_threshold,
    )