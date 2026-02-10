"""Plot training and test step accuracy from multiple CSV files.

Usage:
    python plot_accuracy.py title="My Plot Title" file1.csv label=Warm_Start distinguish=K=10 file2.csv label=Cold_Start distinguish=K=20 ...

Arguments:
    title="<title>"                  - plot title (optional, use quotes if spaces)

Each CSV file should be followed by:
    label=<Warm_Start|Cold_Start>   - determines color family (blue for Warm, red for Cold)
    distinguish=<value>              - text shown directly on the line (e.g., K=10, starvation=5)

Example:
    python plot_accuracy.py title="Step Accuracy Comparison" \
        warm_k10.csv label=Warm_Start distinguish=K=10 \
        warm_k20.csv label=Warm_Start distinguish=K=20 \
        cold_k10.csv label=Cold_Start distinguish=K=10 \
        cold_k20.csv label=Cold_Start distinguish=K=20
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(args):
    """Parse command line arguments into title and list of (filepath, label,
    distinguish) tuples."""
    files = []
    title = "Training and Test Step Accuracy"
    i = 0

    while i < len(args):
        if args[i].startswith("title="):
            title = args[i].split("=", 1)[1]
            i += 1
            continue

        if args[i].endswith(".csv"):
            filepath = args[i]
            label = None
            distinguish = None

            # Look for label= and distinguish= after the csv file
            while (
                i + 1 < len(args)
                and not args[i + 1].endswith(".csv")
                and not args[i + 1].startswith("title=")
            ):
                i += 1
                if args[i].startswith("label="):
                    label = args[i].split("=", 1)[1]
                elif args[i].startswith("distinguish="):
                    distinguish = args[i].split("=", 1)[1]

            if label is None:
                print(f"Warning: No label for {filepath}, defaulting to 'Unknown'")
                label = "Unknown"
            if distinguish is None:
                print(f"Warning: No distinguish for {filepath}, using filename")
                distinguish = filepath

            files.append((filepath, label, distinguish))
        i += 1

    return title, files


def add_line_label(ax, x, y, text, color, offset_idx, total_lines):
    """Add a label directly on the line at a position that avoids overlap."""
    # Pick position along the line based on index to spread labels out
    positions = np.linspace(0.3, 0.8, total_lines)
    pos = positions[offset_idx % len(positions)]

    idx = int(len(x) * pos)
    if idx >= len(x):
        idx = len(x) - 1

    ax.annotate(
        text,
        xy=(x.iloc[idx], y.iloc[idx]),
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8
        ),
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    title, files = parse_args(sys.argv[1:])

    if not files:
        print("No CSV files provided.")
        sys.exit(1)

    # Color schemes: Warm_Start = blues, Cold_Start = reds
    warm_colors = ["#1f77b4", "#4a90d9", "#0d4f8a", "#2e6b9e", "#5999c7", "#1a5a8a"]
    cold_colors = ["#d62728", "#e55a5a", "#a31d1d", "#c44040", "#b22222", "#8b0000"]

    warm_idx = 0
    cold_idx = 0

    fig, ax = plt.subplots(figsize=(12, 8))

    # Track which labels we've added to legend
    legend_handles = {}

    total_lines = len(files)
    line_idx = 0

    for filepath, label, distinguish in files:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Filter to epochs <= 450
        df = df[df["epoch"] <= 450]

        # Assign color based on label
        if "warm" in label.lower():
            color = warm_colors[warm_idx % len(warm_colors)]
            warm_idx += 1
        else:
            color = cold_colors[cold_idx % len(cold_colors)]
            cold_idx += 1

        # Plot train step accuracy (solid line)
        (train_line,) = ax.plot(
            df["epoch"], df["train_step_acc"], color=color, linestyle="-", linewidth=2
        )

        # Plot test step accuracy (dashed line)
        (test_line,) = ax.plot(
            df["epoch"], df["test_step_acc"], color=color, linestyle="--", linewidth=2
        )

        # Add distinguish label directly on the train line
        add_line_label(
            ax,
            df["epoch"],
            df["train_step_acc"],
            distinguish,
            color,
            line_idx,
            total_lines,
        )

        # Add to legend only once per label type
        if label not in legend_handles:
            legend_handles[label] = (train_line, color)

        line_idx += 1

    # Create custom legend for color coding (Warm vs Cold) and line style (Train vs Test)
    from matplotlib.lines import Line2D

    legend_elements = []

    for label, (line, color) in legend_handles.items():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{label} (Train)",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                linewidth=2,
                label=f"{label} (Test)",
            )
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Step Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save and show
    output_file = "step_accuracy_plot.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
