import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def trace_a(path, component):
    data = json.loads(path.read_text())
    prompt = next(iter(data["results"].values()))
    groups = {}

    for run in prompt["runs"]:
        for row in run["decode_steps"]:
            value = row.get(component)

            if value is not None:
                position = row["n_past_after_eval"] - 1
                groups.setdefault(position, []).append(value)

    x = sorted(groups)
    return x, [np.median(groups[n]) for n in x]

def trace(path, component):
    data = json.loads(path.read_text())
    groups = {}

    for row in data["runs"]:
        if component == "core_decode_ms":
            value = row.get(component, row["eval_ms"] + row["sample_ms"])
        else:
            value = row.get(component)

        if value is not None:
            position = row["n_past_after_eval"] - 1
            groups.setdefault(position, []).append(value)

    x = sorted(groups)
    return (x, [np.median(groups[n]) for n in x])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python-a", type=Path, required=True)
    p.add_argument("--native-async", type=Path, required=True)
    p.add_argument("--native-sync", type=Path, required=True)
    p.add_argument("--output", type=Path, default="figures/comparison.png")
    args = p.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout="constrained")

    # Decode API boundary timing
    axes[0].plot(*trace_a(args.python_a, "eval_ms"), label="Python A eval", linewidth=1.1)
    axes[0].plot(*trace(args.native_async, "decode_ms"), label="Native async decode", linewidth=1.1)

    # Where pending GPU work is waited for
    for path, key, label in [
        (args.native_async, "sample_ms", "Native async sample"),
        (args.native_sync, "sync_ms", "Native explicit synchronization"),
        (args.native_sync, "sample_ms", "Native sample after synchronization")
    ]:
        axes[1].plot(*trace(path, key), label=label, linewidth=1.1)

    # Complete core step
    axes[2].plot(*trace_a(args.python_a, "eval_ms"), label="Python A", linewidth=1.1)
    axes[2].plot(*trace(args.native_async, "core_decode_ms"), label="Native async", linewidth=1.1)
    axes[2].plot(*trace(args.native_sync, "core_decode_ms"), label="Native sync", linewidth=1.1)


    titles = [
        "Decode API boundary timing",
        "Where pending GPU work is waited for",
        "Complete core step",
    ]

    for ax, title in zip(axes, titles):
        for n in [256, 512]:
            ax.axvline(n, color="gray", linestyle="--", alpha=0.5)

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("P50 wall time (ms)")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9)

    axes[2].set_xlabel("Internal n_past BEFORE evaluating the next token")

    fig.suptitle("Exp6-B: native replay of Exp6-A tokens (20 runs per mode)", fontweight="bold", fontsize=14)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()