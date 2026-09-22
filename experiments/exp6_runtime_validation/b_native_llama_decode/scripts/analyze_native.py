import argparse
import json
from pathlib import Path
import numpy as np

COMPONENTS = ["remove_ms", "setup_ms", "decode_ms", "eval_ms", "sync_ms", "sample_ms", "core_decode_ms",]

def analyze(path, window=3):
    data = json.loads(path.read_text())
    input_tokens = data["input_tokens"]
    rows = data["runs"]
    runs = {}
    for row in rows:
        runs.setdefault(row["run_id"], []).append(row)

    results = []
    positions = sorted({row["n_past_after_eval"] - 1 for row in rows})
    targets = [input_tokens] + [n for n in positions if n % 256 == 0 and n != input_tokens]

    for target in targets:
        components = {}

        for component in COMPONENTS:
            values, references, deltas, spikes = [], [], [], 0

            for run_rows in runs.values():
                rows_by_position = {row["n_past_after_eval"] - 1: row for row in run_rows}

                if target not in rows_by_position:
                    continue

                def value(row):
                    if component == "core_decode_ms":
                        return row.get(component, row["eval_ms"] + row["sample_ms"])
                    return row.get(component)

                val = value(rows_by_position[target])
                near = [
                    value(rows_by_position[n])
                    for n in range(target - window,  target + window + 1)
                    if (n != target and n in rows_by_position and (n % 256 != 0 or n == input_tokens))
                ]

                near = [v for v in near if v is not None]

                if val is None or not near:
                    continue

                ref = float(np.median(near))

                values.append(val)
                references.append(ref)
                deltas.append(val - ref)

                spikes += int(val > 1.2 * ref and val - ref > 0.8)

            if values:
                components[component] = {
                    "p50_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "local_p50_ms": float(np.median(references)),
                    "paired_delta_p50_ms": float(np.median(deltas)),
                    "paired_delta_mean_ms": float(np.mean(deltas)),
                    "spikes": spikes,
                    "runs": len(values),
                }

        results.append({
            "n_past_before_eval": target,
            "decode_step": target - input_tokens + 1,
            "event": (
                "first decode (separate)"
                if target == input_tokens
                else "periodic boundary"
            ),
            "components": components,
        })

    additive = []
    for target in targets:
        pairs = []

        for run_rows in runs.values():
            rows_by_position = {row["n_past_after_eval"] - 1: row for row in run_rows}
            if target not in rows_by_position:
                continue

            fields = ["remove_ms", "setup_ms", "decode_ms", "sync_ms", "sample_ms",]

            if not all(key in rows_by_position[target] for key in fields):
                continue

            neighbors = [
                rows_by_position[n]
                for n in range(target - window, target + window + 1)
                if (n != target and n in rows_by_position and n % 256 != 0)
            ]

            if not neighbors:
                continue

            pairs.append({
                key : rows_by_position[target][key] - float(np.mean([row[key] for row in neighbors]))
                for key in fields + ["core_decode_ms"]
            })

        if pairs:
            means = {
                key: float(np.mean([pair[key] for pair in pairs]))
                for key in pairs[0]
            }

            residual = (means["core_decode_ms"] - sum(means[key] for key in fields))

            additive.append({
                "n_past_before_eval": target,
                "paired_mean_deltas_ms": means,
                "residual_ms": residual,
            })

    return {
        "source": str(path),
        "mode": data.get("mode"),
        "runs": len(runs),
        "input_tokens": input_tokens,
        "decode_steps": data.get("decode_steps"),
        "rule": (
            "target > local median * 1.2 "
            "AND excess > 0.8 ms"
        ),
        "window": window,
        "targets": results,
        "additive_decomposition": additive,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path, help="Native Exp6-B JSON",)
    p.add_argument("--compare", type=Path, help="Optional Exp6-A JSON for comparison",)
    p.add_argument("--output", type=Path, required=True, help="Analysis JSON output",)
    p.add_argument("--window", type=int, default=3, help="Local reference window",)
    args = p.parse_args()

    result = {"native": analyze(args.native, args.window,)}

    if args.compare:
        result["python_a"] = analyze(args.compare, args.window,)

    args.output.write_text(json.dumps(result, indent=2,))

    for label, report in result.items():
        print(label, report["runs"], "runs")

        for target in report["targets"]:
            print("n_past", target["n_past_before_eval"], target["event"],)

            for component, value in target["components"].items():
                print(
                    f"  {component:16s} "
                    f"P50={value['p50_ms']:.3f} "
                    f"local={value['local_p50_ms']:.3f} "
                    f"paired delta="
                    f"{value['paired_delta_p50_ms']:+.3f} "
                    f"spikes="
                    f"{value['spikes']}/{value['runs']}"
                )


if __name__ == "__main__":
    main()