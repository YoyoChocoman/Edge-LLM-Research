"""Align buffered events; keep unobserved states distinct from false."""
from collections import defaultdict

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
        pads = kinds["padding"]

        if len(pads) != 1:
            raise ValueError(f"Expected one effective-padding event at {key}")

        pad = pads[0]
        if pad["effective_padding"] != timing["padding"] or pad["n_kv"] != kv["n_kv"]:
            raise ValueError(f"Runtime padding does not match this group's label at {key}")

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
            "required_padding": pad["required_padding"], "effective_padding": pad["effective_padding"],
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
