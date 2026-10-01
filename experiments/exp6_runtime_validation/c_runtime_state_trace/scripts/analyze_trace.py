"""Align runtime events with token timings; summarize boundaries without assigning causes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

METRICS = ["eval_ms", "decode_ms", "sync_ms", "sample_ms", "core_decode_ms"]
PERIOD = 256
WINDOW = 3
SPIKE_RATIO = 1.2
SPIKE_DELTA_MS = 0.8


def local_values(rows, index, metric, window):
    target = rows[index]["n_past_before_eval"]
    return [r[metric] for r in rows
            if 0 < abs(r["n_past_before_eval"] - target) <= window
            and r["n_past_before_eval"] % PERIOD != 0 and r["decode_step"] != 1]


def timing_summary(rows, position, window):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["run_id"]].append(row)

    result = {}
    for metric in METRICS:
        values, local, deltas, spikes = [], [], [], 0

        for run in grouped.values():
            run.sort(key=lambda r: r["decode_step"])

            for i, row in enumerate(run):
                if row["n_past_before_eval"] != position:
                    continue

                values.append(row[metric])
                neighbors = local_values(run, i, metric, window)

                if neighbors:
                    reference = median(neighbors)
                    local.append(reference)
                    deltas.append(row[metric] - reference)
                    spikes += row[metric] > SPIKE_RATIO * reference and row[metric] - reference > SPIKE_DELTA_MS

        result[metric] = {
            "p50_ms": median(values) if values else None,
            "local_p50_ms": median(local) if local else None,
            "paired_delta_p50_ms": median(deltas) if deltas else None,
            "spikes": spikes, "compared_runs": len(deltas), "runs": len(values),
        }

    return result


def align(timing, trace):
    if trace["dropped_events"]:
        raise ValueError("Trace buffer overflow: increase the capacity in native.cpp and rerun")

    events = defaultdict(list)
    keys = {(r["run_id"], r["decode_step"], r["n_past_before_eval"]) for r in timing["runs"]}

    for event in trace["events"]:
        key = (event["run_id"], event["decode_step"], event["n_past_before_eval"])

        if key not in keys:
            raise ValueError("Events and timings do not belong to the same replay")

        events[key].append(event)

    aligned = []
    previous = {}
    for row in sorted(timing["runs"], key=lambda r: (r["run_id"], r["decode_step"])):
        key = (row["run_id"], row["decode_step"], row["n_past_before_eval"])
        kinds = defaultdict(list)

        for event in events[key]:
            kinds[event["kind"]].append(event)

        # This experiment uses one token and one sequence per llama_decode.
        if len(kinds["kv"]) != 1 or len(kinds["graph"]) != 1:
            raise ValueError(f"Expected one KV and graph event per step; inspect events.json at {key}")

        kv, graph = kinds["kv"][0], kinds["graph"][0]
        builds, checks = kinds["mask_build"], kinds["mask_check"]
        shapes = [[e[f"ne{i}"] for i in range(4)] for e in builds]

        if not shapes:
            shapes = [[e[f"new{i}"] for i in range(4)] for e in checks if e["compatible"]]

        shapes = [list(s) for s in sorted(set(tuple(s) for s in shapes))] or None
        cuda = kinds["cuda"]
        cuda_modes = ["capture" if e["capture"] else "reuse" if e["use_graph"] else "direct" for e in cuda]
        old = previous.get(row["run_id"])
        state = {
            **row,
            "n_kv": kv["n_kv"], "kv_capacity": kv["kv_capacity"],
            "n_kv_changed": kv["n_kv"] != old["n_kv"] if old else None,
            "mask_shapes": shapes,
            "mask_changed": shapes != old["mask_shapes"] if old and shapes and old["mask_shapes"] else None,
            "mask_check_failed": any(not e["compatible"] for e in checks) if checks else None,
            "graph_reused": bool(graph["reused"]),
            "graph_reuse_disabled": bool(graph["reuse_disabled"]),
            "cuda_modes": cuda_modes,
            "cuda_capture": any(e["capture"] for e in cuda) if cuda else None,
            "cuda_direct": any(not e["use_graph"] for e in cuda) if cuda else None,
            "cuda_modes_changed": cuda_modes != old["cuda_modes"] if old and cuda_modes and old["cuda_modes"] else None,
        }
        aligned.append(state)
        previous[row["run_id"]] = state

    return aligned


def rate(rows, field):
    known = [r[field] for r in rows if r[field] is not None]
    return {"count": sum(known), "observed": len(known)}


def check_baseline(timing, baseline):
    if baseline.get("variant") != "baseline" or baseline["mode"] != timing["mode"]:
        raise ValueError("Use a C baseline from the same sync/async mode")

    def sequence(data):
        first = min(r["run_id"] for r in data["runs"])
        return sorted((r["decode_step"], r["n_past_before_eval"], r["evaluated_token_id"], r["expected_next_token_id"])
                      for r in data["runs"] if r["run_id"] == first)
    if timing["input_tokens"] != baseline["input_tokens"] or sequence(timing) != sequence(baseline):
        raise ValueError("Baseline and trace replay different token sequences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path, help="Traced run's native.json; events.json is read from the same folder")
    parser.add_argument("--baseline", type=Path, help="Optional rebuilt, uninstrumented native.json")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--output", type=Path, help="Default: analysis.json next to native.json")
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be positive")

    timing = json.loads(args.native.read_text())
    if timing.get("variant") != "trace":
        parser.error("native must be a trace run")

    trace = json.loads(args.native.with_name("events.json").read_text())
    rows = align(timing, trace)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline:
        check_baseline(timing, baseline)

    first = timing["input_tokens"]
    targets = sorted({r["n_past_before_eval"] for r in rows
                      if r["decode_step"] == 1 or r["n_past_before_eval"] % PERIOD == 0})

    summaries = []
    for position in targets:
        selected = [r for r in rows if r["n_past_before_eval"] == position]
        summary = {
            "n_past_before_eval": position,
            "kind": "first_decode" if position == first else "periodic_boundary",
            "trace_timing": timing_summary(rows, position, args.window),
            "states": {field: rate(selected, field) for field in
                       ["n_kv_changed", "mask_changed", "mask_check_failed", "graph_reused", "cuda_capture", "cuda_direct"]},
        }

        if baseline:
            summary["baseline_timing"] = timing_summary(baseline["runs"], position, args.window)
        summaries.append(summary)

    transitions = [r for r in rows if r["decode_step"] != 1 and
                   (r["n_kv_changed"] or r["mask_changed"] or not r["graph_reused"] or r["cuda_modes_changed"])]

    report = {
        "mode": timing["mode"], "input_tokens": first, "window": args.window,
        "spike_rule": {"ratio": SPIKE_RATIO, "delta_ms": SPIKE_DELTA_MS},
        "native_file": str(args.native), "baseline_file": str(args.baseline) if baseline else None,
        "targets": summaries, "transitions": transitions, "steps": rows,
    }

    output = args.output or args.native.with_name("analysis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("position  eval p50  paired delta  KV change  mask change  graph reuse  CUDA capture")

    for summary in summaries:
        states = summary["states"]
        counts = [f"{states[k]['count']}/{states[k]['observed']}" for k in
                  ["n_kv_changed", "mask_changed", "graph_reused", "cuda_capture"]]
        metric = summary["trace_timing"]["eval_ms"]
        print(summary["n_past_before_eval"], metric["p50_ms"], metric["paired_delta_p50_ms"], *counts)

    print(f"Analysis: {output}")


if __name__ == "__main__":
    main()
