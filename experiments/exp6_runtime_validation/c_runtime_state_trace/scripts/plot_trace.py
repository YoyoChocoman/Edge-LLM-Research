"""Plot timing and observed states for one run around the periodic boundaries."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

WINDOW = 4
TARGETS = [256, 512]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--targets", type=int, nargs="+", default=TARGETS)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    rows = [r for r in report["steps"] if r["run_id"] == args.run]
    if not rows:
        parser.error("Selected run is absent")
    fig, axes = plt.subplots(4, len(args.targets), figsize=(6 * len(args.targets), 10), squeeze=False, sharex="col")
    for col, target in enumerate(args.targets):
        selected = [r for r in rows if abs(r["n_past_before_eval"] - target) <= args.window]
        if not selected:
            parser.error(f"No samples near {target}")
        x = [r["n_past_before_eval"] for r in selected]
        for metric in ["eval_ms", "core_decode_ms"]:
            axes[0, col].plot(x, [r[metric] for r in selected], "o-", label=metric)
        axes[0, col].set_title(f"n_past before eval = {target}")
        axes[0, col].set_ylabel("Latency (ms)")
        axes[0, col].legend()
        axes[1, col].step(x, [r["n_kv"] for r in selected], where="mid", label="n_kv")
        # Plot the actual mask's first dimension, not a value inferred from n_kv.
        mask = [r["mask_shapes"][0][0] if r["mask_shapes"] and len(r["mask_shapes"]) == 1 else float("nan") for r in selected]
        axes[1, col].plot(x, mask, "x", markersize=8, label="mask ne[0]")
        axes[1, col].set_ylabel("KV / mask extent")
        axes[1, col].legend()
        axes[2, col].plot(x, [int(r["graph_reused"]) for r in selected], "o-")
        axes[2, col].set_yticks([0, 1], ["build", "reuse"])
        axes[2, col].set_ylim(-0.3, 1.3)
        axes[2, col].set_ylabel("llama graph")
        for label, level in [("direct", 0), ("capture", 1), ("reuse", 2)]:
            positions = [r["n_past_before_eval"] for r in selected if label in r["cuda_modes"]]
            axes[3, col].scatter(positions, [level] * len(positions), label=label)
        missing = [r["n_past_before_eval"] for r in selected if not r["cuda_modes"]]
        axes[3, col].scatter(missing, [-1] * len(missing), marker="x", color="gray")
        axes[3, col].set_yticks([-1, 0, 1, 2], ["unobserved", "direct", "capture", "reuse"])
        axes[3, col].set_ylim(-1.3, 2.3)
        axes[3, col].set_ylabel("CUDA Graph")
        axes[3, col].set_xlabel("n_past_before_eval")
        axes[3, col].set_xticks(x)
        for ax in axes[:, col]:
            ax.axvline(target, color="gray", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.2)
    fig.suptitle(f"Exp6-C: observed runtime states | {report['mode']} | run {args.run}")
    fig.tight_layout()
    output = args.output or Path("figures") / f"{args.analysis.parent.name}_run{args.run}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
