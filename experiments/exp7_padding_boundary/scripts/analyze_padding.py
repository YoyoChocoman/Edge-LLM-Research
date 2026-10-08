"""Compare observed transition positions and two-step latency windows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from trace_events import align

TARGETS = [256, 512, 768, 1024]
WINDOW = 3
METRICS = ["decode_ms", "core_decode_ms"]
SPIKE_RATIO = 1.2
SPIKE_DELTA_MS = 0.8

def load_group(paths, padding):
    steps, batches = [], []
    reference = None
    next_run = 0

    for path in sorted(paths):
        data = json.loads(path.read_text())
        trace = json.loads(path.with_name("events.json").read_text())
        aligned = align(data, trace)
        grouped = defaultdict(list)

        for row in aligned:
            grouped[row["run_id"]].append(row)

        for local_run, rows in sorted(grouped.items()):
            rows.sort(key=lambda r: r["decode_step"])

            signature = {
                "prompt_tokens": data["prompt_tokens"], "context": data["context"],
                "warmup_steps": data["warmup_steps"], "input_tokens": data["input_tokens"],
                "sequence": [(r["evaluated_token_id"], r["expected_next_token_id"]) for r in rows],
            }

            reference = signature
            next_run += 1

            for row in rows:
                row.update(run_id=next_run, local_run_id=local_run, batch=str(path.parent))
                steps.append(row)

        batches.append({"native_file": str(path), "runs": len(grouped)})
    return steps, batches, reference

def is_event(row):
    return bool(row["n_kv_changed"] or row["mask_changed"] or not row["graph_reused"]
                or row["cuda_capture"] or row["cuda_direct"])

def timing(rows, position, excluded, window):
    by_run = defaultdict(dict)

    for row in rows:
        by_run[row["run_id"]][row["n_past_before_eval"]] = row

    result = {}
    for metric in METRICS:
        values, following, references, deltas, sums, sum_deltas = [], [], [], [], [], []
        spikes = 0

        for run in by_run.values():
            current, after = run[position], run[position + 1]
            values.append(current[metric])
            following.append(after[metric])
            sums.append(current[metric] + after[metric])
            neighbors = [r[metric] for p, r in run.items() if position - window <= p <= position + 1 + window and p not in excluded]

            if neighbors:
                local = median(neighbors)
                references.append(local)
                deltas.append(current[metric] - local)
                sum_deltas.append(current[metric] + after[metric] - 2 * local)
                spikes += current[metric] > SPIKE_RATIO * local and current[metric] - local > SPIKE_DELTA_MS

        result[metric] = {
            "p50_ms": median(values), "next_step_p50_ms": median(following),
            "local_p50_ms": median(references) if references else None,
            "paired_excess_p50_ms": median(deltas) if deltas else None,
            "two_step_p50_ms": median(sums),
            "two_step_excess_p50_ms": median(sum_deltas) if sum_deltas else None,
            "spikes": spikes, "compared_runs": len(references), "runs": len(values),
        }

    return result

def rate(rows, field):
    observed = [r[field] for r in rows if r[field] is not None]
    return {"count": sum(observed), "observed": len(observed)}

def summarize(rows, targets, excluded, window, padding):
    result = []
    for position in targets:
        current = [r for r in rows if r["n_past_before_eval"] == position]
        after = [r for r in rows if r["n_past_before_eval"] == position + 1]
        result.append({
            "n_past_before_eval": position,
            "predicted_transition": position % padding == 0,
            "n_kv_observed": sorted({r["n_kv"] for r in current}),
            "states": {field: rate(current, field) for field in
                       ["n_kv_changed", "mask_changed", "mask_check_failed", "graph_reused", "cuda_direct"]},
            "next_step_capture": rate(after, "cuda_capture"),
            "timing": timing(rows, position, excluded, window),
        })
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad256", type=Path, nargs="+", required=True, help="One or more native.json files")
    parser.add_argument("--pad512", type=Path, nargs="+", required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=TARGETS)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--output", type=Path, default="results/comparison.json")
    args = parser.parse_args()

    paths = args.pad256 + args.pad512
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("The same file was supplied more than once")

    groups = {}
    reference = None
    for padding, paths in [(256, args.pad256), (512, args.pad512)]:
        rows, batches, signature = load_group(paths, padding)
        if reference is not None and signature != reference:
            raise ValueError("Padding groups do not share the same replay and context settings")

        reference = signature
        groups[str(padding)] = {"steps": rows, "batches": batches}

    # Use the same stable reference positions for both groups, including when an old event disappears.
    excluded = {p + offset for p in args.targets for offset in [0, 1]}
    for group in groups.values():
        excluded.update(r["n_past_before_eval"] for r in group["steps"] if r["decode_step"] <= 2 or is_event(r))

    sampled = {}
    for padding, group in groups.items():
        rows = group["steps"]
        group["targets"] = summarize(rows, args.targets, excluded, args.window, int(padding))
        group["transitions"] = [r for r in rows if r["decode_step"] > 2 and is_event(r)]
        group["runs"] = len({r["run_id"] for r in rows})
        group["stable_core_p50_ms"] = median(r["core_decode_ms"] for r in rows if r["n_past_before_eval"] not in excluded)
        totals = defaultdict(float)
        outputs = defaultdict(set)

        for row in rows:
            totals[row["run_id"]] += row["core_decode_ms"]
            outputs[row["decode_step"]].add(row["sampled_next_token_id"])

        group["total_core_p50_ms"] = median(totals.values())
        group["expected_token_mismatches"] = sum(r["sampled_next_token_id"] != r["expected_next_token_id"] for r in rows)
        group["within_group_sample_variation_steps"] = [step for step, values in outputs.items() if len(values) > 1]
        sampled[padding] = outputs
        group["padding_extent_mismatches"] = sum(
            r["n_kv"] != min(r["kv_capacity"], ((r["n_past_before_eval"] + 1 + int(padding) - 1) // int(padding)) * int(padding))
            for r in rows)

        for target in group["targets"]:
            m = target["timing"]["decode_ms"]
            print(f"pad={padding} position={target['n_past_before_eval']} "
                  f"decode={m['p50_ms']:.3f} ms two-step excess={m['two_step_excess_p50_ms']} "
                  f"KV changes={target['states']['n_kv_changed']}")

    differences = [
        {"decode_step": step, "pad256": sorted(values), "pad512": sorted(sampled["512"][step])}
        for step, values in sampled["256"].items() if values != sampled["512"][step]
    ]

    report = {
        "window": args.window, "targets": args.targets, "excluded_reference_positions": sorted(excluded),
        "spike_rule": {"ratio": SPIKE_RATIO, "delta_ms": SPIKE_DELTA_MS},
        "sampled_token_differences": differences, "groups": groups,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Sampled-token set differences: {len(differences)} positions")
    print(f"Analysis: {args.output}")

if __name__ == "__main__":
    main()
