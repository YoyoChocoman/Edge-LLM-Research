import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = {
    "eval_ms": "llm.eval() Latency",
    "sample_ms": "llm.sample() Latency",
    "core_decode_ms": "Core Decode (eval + sample)",
}


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def load_component_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompt_name, prompt_data = next(iter(data["results"].items()))

    values = {
        component: {}
        for component in COMPONENTS
    }

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

    return values


def plot_components(values, target_steps, output_path):

    max_step = max(
        step
        for component_values in values.values()
        for step in component_values
    )

    steps = np.arange(1, max_step + 1)

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 10),
        sharex=True,
    )

    colors = {
        "P50": "blue",
        "P90": "orange",
        "P95": "red",
    }

    for axis, (component, label) in zip(
        axes,
        COMPONENTS.items()
    ):

        # =====================================================
        # 1. RAW DATA
        # =====================================================

        raw_x = []
        raw_y = []

        for step in steps:

            step_values = values[component].get(step, [])

            for value in step_values:
                raw_x.append(step)
                raw_y.append(value)

        # Raw measurements shown as gray dots
        axis.scatter(
            raw_x,
            raw_y,
            s=10,
            color="gray",
            alpha=0.35,
            edgecolors="none",
            label="Raw Data",
            zorder=1,
        )

        # =====================================================
        # 2. Percentiles
        # =====================================================

        p50 = [
            percentile(
                values[component].get(step, []),
                50
            )
            for step in steps
        ]

        p90 = [
            percentile(
                values[component].get(step, []),
                90
            )
            for step in steps
        ]

        p95 = [
            percentile(
                values[component].get(step, []),
                95
            )
            for step in steps
        ]

        # P50
        axis.plot(
            steps,
            p50,
            label="P50",
            color=colors["P50"],
            linewidth=2,
            zorder=3,
        )

        # P90
        axis.plot(
            steps,
            p90,
            label="P90",
            color=colors["P90"],
            linestyle="--",
            linewidth=1.5,
            zorder=3,
        )

        # P95
        axis.plot(
            steps,
            p95,
            label="P95",
            color=colors["P95"],
            linestyle=":",
            linewidth=1.5,
            zorder=3,
        )

        # =====================================================
        # 3. Target Steps
        # =====================================================

        for target in target_steps:

            axis.axvline(
                target,
                color="gray",
                alpha=0.6,
                linestyle="-.",
                linewidth=1,
                label=(
                    f"Target Step {target}"
                    if component == "eval_ms"
                    else ""
                ),
                zorder=2,
            )

        # =====================================================
        # 4. Y-axis Scaling
        # =====================================================

        # Fix sample latency scale to 4–10 ms
        if component == "sample_ms":
            axis.set_ylim(4, 10)

        # =====================================================
        # 5. Formatting
        # =====================================================

        axis.set_title(
            label,
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        axis.set_ylabel(
            "Latency (ms)",
            fontsize=10,
        )

        axis.grid(
            True,
            linestyle="--",
            alpha=0.4,
            zorder=0,
        )

        axis.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
        )

    # ---------------------------------------------------------
    # X axis
    # ---------------------------------------------------------

    axes[-1].set_xlabel(
        "Decode Step (Position)",
        fontsize=11,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # Overall title
    # ---------------------------------------------------------

    fig.suptitle(
        "Exp6-A: Decode-Time Component Decomposition & Dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.93]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"\nSaved professional plot to: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot eval/sample timing components "
            "from Exp6-A JSON."
        )
    )

    parser.add_argument(
        "json_file",
        help="Path to Exp6-A JSON result",
    )

    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[1, 134, 390],
        help="Decode steps to mark",
    )

    parser.add_argument(
        "--output",
        default="figures/component.png",
        help="Output plot path",
    )

    args = parser.parse_args()

    values = load_component_data(
        args.json_file
    )

    plot_components(
        values,
        args.targets,
        Path(args.output),
    )


if __name__ == "__main__":
    main()