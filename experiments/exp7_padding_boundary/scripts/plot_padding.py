"""Plot the full latency trajectory and one observed run's boundary states."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt

COLORS = {"256": "tab:blue", "512": "tab:orange"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--run", type=int, default=1, help="Aggregate run index within each group")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default="figures")
    args = parser.parse_args()

    report = json.loads(args.analysis.read_text())
    targets = report["targets"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    for padding, group in report["groups"].items():
        by_position = defaultdict(list)

        for row in group["steps"]:
            if row["decode_step"] > 2:
                by_position[row["n_past_before_eval"]].append(row)

        x = sorted(by_position)
        for ax, metric in zip(axes, ["decode_ms", "core_decode_ms"]):
            ax.plot(x, [median(r[metric] for r in by_position[p]) for p in x],
                    color=COLORS[padding], label=f"padding {padding}", linewidth=1)
            ax.set_ylabel(f"{metric} P50")

    for ax in axes:
        for target in targets:
            ax.axvline(target, color="gray", linestyle="--", alpha=0.35)

        ax.grid(alpha=0.2)
        ax.legend()

    axes[-1].set_xlabel("n_past_before_eval")
    fig.suptitle("Exp7: fixed replay, padding 256 vs 512 (first two steps excluded)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "latency_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, len(targets), figsize=(5 * len(targets), 11), sharex="col", squeeze=False)
    for padding, group in report["groups"].items():
        rows = [r for r in group["steps"] if r["run_id"] == args.run]
        if not rows:
            parser.error(f"Run {args.run} is absent in padding {padding}")

        for col, target in enumerate(targets):
            selected = [r for r in rows if abs(r["n_past_before_eval"] - target) <= args.window]
            if not selected:
                parser.error(f"No observations near {target}")

            x = [r["n_past_before_eval"] for r in selected]
            color = COLORS[padding]
            axes[0, col].plot(x, [r["decode_ms"] for r in selected], "o-", color=color, label=f"pad {padding}")
            axes[1, col].step(x, [r["n_kv"] for r in selected], where="mid", color=color)
            shapes = [r["mask_shapes"][0][0] if r["mask_shapes"] and len(r["mask_shapes"]) == 1 else float("nan") for r in selected]
            axes[1, col].plot(x, shapes, "x", color=color, markersize=8)
            offset = -0.07 if padding == "256" else 0.07
            axes[2, col].plot(x, [int(r["graph_reused"]) + offset for r in selected], "o-", color=color)

            for row in selected:
                modes = set(row["cuda_modes"]) or {"unobserved"}
                for mode in modes:
                    level = {"unobserved": -1, "direct": 0, "capture": 1, "reuse": 2}[mode]
                    axes[3, col].scatter(row["n_past_before_eval"], level + offset, color=color, s=25)

    for col, target in enumerate(targets):
        axes[0, col].set_title(f"Before-position {target}")
        axes[0, col].set_ylabel("Decode (ms)")
        axes[0, col].legend()
        axes[1, col].set_ylabel("n_kv (line) / mask ne[0] (x)")
        axes[2, col].set_yticks([0, 1], ["build", "reuse"])
        axes[2, col].set_ylim(-0.3, 1.3)
        axes[2, col].set_ylabel("llama graph")
        axes[3, col].set_yticks([-1, 0, 1, 2], ["unobserved", "direct", "capture", "reuse"])
        axes[3, col].set_ylim(-1.3, 2.3)
        axes[3, col].set_ylabel("CUDA Graph")
        axes[3, col].set_xlabel("n_past_before_eval")

        for ax in axes[:, col]:
            ax.axvline(target, color="gray", linestyle="--", alpha=0.4)
            ax.grid(alpha=0.2)

    fig.suptitle(f"Exp7: observed boundary states, aggregate run {args.run} per group (not paired runs)")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"boundary_states_run{args.run}.png", dpi=180)
    plt.close(fig)
    print(f"Figures: {args.output_dir}")

if __name__ == "__main__":
    main()
