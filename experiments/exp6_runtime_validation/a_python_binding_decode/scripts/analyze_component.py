import argparse
import json
import numpy as np

COMPONENTS = {
    "eval_ms": "llm.eval()",
    "sample_ms": "llm.sample()",
    "core_decode_ms": "eval + sample",
}

def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")

def local_median(values_by_step, target_step, window):
    neighbors = []
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        step = target_step + offset
        neighbors.extend(values_by_step.get(step, []))
    return float(np.median(neighbors)) if neighbors else None

def load_component_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompt_name, prompt_data = next(iter(data["results"].items()))
    values = {component: {} for component in COMPONENTS}
    positions = {}

    for run in prompt_data["runs"]:
        for item in run["decode_steps"]:
            step = item["decode_step"]
            eval_ms = item["eval_ms"]
            sample_ms = item["sample_ms"]

            row = {
                "eval_ms": eval_ms,
                "sample_ms": sample_ms,
                "core_decode_ms": eval_ms + sample_ms,
            }

            for component, value in row.items():
                values[component].setdefault(step, []).append(value)

            n_after = item.get("n_past_after_eval")
            if n_after is not None:
                positions.setdefault(step, []).append(n_after - 1)

    return data["metadata"], prompt_name, prompt_data, values, positions

def print_target_summary(values, positions, target_steps, window):
    print("=" * 100)
    print("EXP6-A COMPONENT DECOMPOSITION ANALYSIS")
    print("=" * 100)

    for target_step in target_steps:
        print(f"\nTarget decode step: {target_step}")

        if target_step in positions:
            print(
                "Internal n_past before eval: "
                f"{np.median(positions[target_step]):.0f}"
            )

        for component, label in COMPONENTS.items():
            target_values = values[component].get(target_step, [])
            baseline = local_median(values[component], target_step, window)

            if not target_values or baseline is None:
                print(f"  {label:<18} unavailable")
                continue

            p50 = percentile(target_values, 50)
            p90 = percentile(target_values, 90)
            p95 = percentile(target_values, 95)
            delta = p50 - baseline
            ratio = p50 / baseline if baseline > 0 else float("nan")

            print(
                f"  {label:<18} "
                f"P50={p50:>7.3f} ms | "
                f"P95={p95:>7.3f} ms | "
                f"local={baseline:>7.3f} ms | "
                f"Δ={delta:>+7.3f} ms | "
                f"ratio={ratio:>5.2f}×"
            )
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description="Analyze eval/sample timing components from Exp6-A JSON.")
    parser.add_argument("json_file", help="Path to Exp6-A JSON result")
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 134, 390], help="Decode steps to inspect")
    parser.add_argument("--window", type=int, default=3, help="Neighboring steps used as local baseline")
    args = parser.parse_args()

    metadata, prompt_name, prompt_data, values, positions = load_component_data(args.json_file)

    print(f"Experiment: {metadata.get('experiment', 'N/A')}")
    print(f"Prompt: {prompt_name}")
    print(f"Input tokens: {prompt_data.get('input_tokens', 'N/A')}")
    print_target_summary(values, positions, args.targets, args.window)

if __name__ == "__main__":
    main()