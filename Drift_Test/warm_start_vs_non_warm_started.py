#!/usr/bin/env python3
"""Plot arbiter test-step accuracy comparing random initialization vs warm-started
models.

Supports multiple trial runs and averages across them. Includes time-to-threshold
analysis for comparing learning speed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


def load_json(filepath: str) -> list:
    """Load JSON data from file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_epochs_and_accuracy(data: list, max_epoch: int = 950) -> dict:
    """Extract epoch and test_step accuracy for each arbiter in the dataset."""
    results = {}
    for entry in data:
        tlsf_file = entry.get("tlsf_file", "unknown")
        arbiter_name = Path(tlsf_file).stem
        epochs = entry.get("epochs", [])

        epoch_nums = [e["epoch"] for e in epochs if e["epoch"] <= max_epoch]
        test_step_acc = [e["test_step"] for e in epochs if e["epoch"] <= max_epoch]

        results[arbiter_name] = {"epochs": epoch_nums, "test_step": test_step_acc}
    return results


def load_and_average_trials(filepaths: list, max_epoch: int = 950) -> dict:
    """Load multiple trial files and average the results per arbiter."""
    all_trials = defaultdict(list)

    for filepath in filepaths:
        data = load_json(filepath)
        trial_data = extract_epochs_and_accuracy(data, max_epoch)

        for arbiter_name, values in trial_data.items():
            all_trials[arbiter_name].append(values)

    averaged = {}
    for arbiter_name, trials in all_trials.items():
        if not trials:
            continue

        epoch_to_values = defaultdict(list)
        for trial in trials:
            for epoch, acc in zip(trial["epochs"], trial["test_step"]):
                epoch_to_values[epoch].append(acc)

        common_epochs = sorted(
            e for e, v in epoch_to_values.items() if len(v) == len(trials)
        )

        if not common_epochs:
            common_epochs = sorted(epoch_to_values.keys())

        mean_acc = [np.mean(epoch_to_values[e]) for e in common_epochs]
        std_acc = [np.std(epoch_to_values[e]) for e in common_epochs]

        averaged[arbiter_name] = {
            "epochs": common_epochs,
            "test_step": mean_acc,
            "test_step_std": std_acc,
            "n_trials": len(trials),
        }

    return averaged


# =============================================================================
# Time-to-Threshold Analysis
# =============================================================================


def epochs_to_threshold(epochs, accuracies, threshold=0.9):
    for epoch, acc in zip(epochs, accuracies):
        if acc >= threshold:
            return epoch
    return None


def compute_threshold_distribution(data, threshold=0.9):
    results = []
    for arbiter_name, values in data.items():
        ttt = epochs_to_threshold(values["epochs"], values["test_step"], threshold)
        final_acc = values["test_step"][-1] if values["test_step"] else None
        results.append(
            {
                "arbiter": arbiter_name,
                "epochs_to_threshold": ttt,
                "reached_threshold": ttt is not None,
                "final_accuracy": final_acc,
            }
        )
    return results


def summarize_distribution(threshold_results, label="Dataset", threshold=0.9):
    reached = [
        r["epochs_to_threshold"] for r in threshold_results if r["reached_threshold"]
    ]
    not_reached = [r for r in threshold_results if not r["reached_threshold"]]

    print(f"\n{'='*60}")
    print(f" {label} (threshold = {threshold})")
    print(f"{'='*60}")
    print(f"Total arbiters: {len(threshold_results)}")
    print(
        f"Reached threshold: {len(reached)} ({100*len(reached)/len(threshold_results):.1f}%)"
    )
    print(
        f"Did not reach: {len(not_reached)} ({100*len(not_reached)/len(threshold_results):.1f}%)"
    )

    if reached:
        arr = np.array(reached)
        print("\nEpochs to threshold (for arbiters that reached it):")
        print(f"  Mean:   {arr.mean():.1f}")
        print(f"  Std:    {arr.std():.1f}")
        print(f"  Median: {np.median(arr):.1f}")
        print(f"  Min:    {arr.min()}")
        print(f"  Max:    {arr.max()}")
        print(f"  25th percentile: {np.percentile(arr, 25):.1f}")
        print(f"  75th percentile: {np.percentile(arr, 75):.1f}")

    if not_reached:
        print("\nArbiters that didn't reach threshold:")
        for r in not_reached:
            acc = r["final_accuracy"]
            acc_str = f"{acc:.4f}" if acc is not None else "—"
            print(f"  {r['arbiter']}: final acc = {acc_str}")

    return reached


def compare_distributions(
    times_a, times_b, label_a="Random Init", label_b="Warm Started", threshold=0.9
):

    print(f"\n{'='*60}")
    print(f" Comparison: {label_a} vs {label_b} (threshold = {threshold})")
    print(f"{'='*60}")

    if not times_a or not times_b:
        print("Cannot compare: one or both sets empty")
        return None

    a, b = np.array(times_a), np.array(times_b)
    diff = a.mean() - b.mean()

    print("\nMean epochs to threshold:")
    print(f"  {label_a}: {a.mean():.1f} ± {a.std():.1f}")
    print(f"  {label_b}: {b.mean():.1f} ± {b.std():.1f}")
    print(f"  Difference: {diff:.1f} epochs")

    if diff > 0:
        print(f"  → {label_b} learns faster by {abs(diff):.1f} epochs")
    else:
        print(f"  → {label_a} learns faster by {abs(diff):.1f} epochs")

    try:
        from scipy import stats

        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print("\nMann-Whitney U test:")
        print(f"  U-statistic: {stat:.1f}")
        print(f"  p-value: {p:.4f}")

        effect_size = 1 - (2 * stat) / (len(a) * len(b))
        print(f"  Effect size (rank-biserial r): {effect_size:.3f}")

        return {
            "u_stat": stat,
            "p_value": p,
            "effect_size": effect_size,
            "mean_diff": diff,
        }

    except ImportError:
        print("\n(scipy not available)")
        return {"mean_diff": diff}


# =============================================================================
# Plotting
# =============================================================================


def plot_comparison(
    random_data, warm_data, output_path=None, show_std=True, n_charts=3
):

    common_arbiters = sorted(set(random_data) & set(warm_data))
    if not common_arbiters:
        print("No common arbiters")
        return

    print(f"Plotting {len(common_arbiters)} arbiters across {n_charts} separate files")

    # Fixed group for chart 1 - ONLY these 5 arbiters
    chart1_names = [
        "Arbiter_N_4",
        "Arbiter_N_5",
        "Arbiter_with_Cancel_N_2",
        "Arbiter_with_Cancel_N_3",
        "Round_Robin_Arbiter_N_4",
    ]

    # Build case-insensitive lookup
    common_lower = {a.lower().replace(" ", "_"): a for a in common_arbiters}

    chart1_arbiters = []
    for name in chart1_names:
        key = name.lower().replace(" ", "_")
        if key in common_lower:
            chart1_arbiters.append(common_lower[key])

    print(f"Chart 1 arbiters: {chart1_arbiters}")

    # Remaining arbiters for other charts
    remaining = [a for a in common_arbiters if a not in chart1_arbiters]
    print(f"Remaining arbiters: {remaining}")

    # Split remaining evenly across remaining charts
    def split(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]

    if n_charts > 1:
        other_groups = split(remaining, n_charts - 1)
        groups = [chart1_arbiters] + other_groups
    else:
        groups = [chart1_arbiters + remaining]

    colors = plt.cm.tab10.colors

    # Determine output base name
    if output_path:
        base = Path(output_path)
        stem = base.stem
        suffix = base.suffix or ".png"
        parent = base.parent
    else:
        stem = "comparison"
        suffix = ".png"
        parent = Path(".")

    # Get trial counts for annotation
    first_arb = common_arbiters[0]
    n_random = random_data[first_arb].get("n_trials", 1)
    n_warm = warm_data[first_arb].get("n_trials", 1)

    for chart_idx, group in enumerate(groups):
        if not group:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        arbiter_colors = {}

        for i, arb in enumerate(group):
            c = colors[i % len(colors)]
            arbiter_colors[arb] = c
            r, w = random_data[arb], warm_data[arb]

            # Random init - dashed line
            ax.plot(r["epochs"], r["test_step"], "--", color=c, lw=2, alpha=0.8)
            # Warm started - solid line
            ax.plot(w["epochs"], w["test_step"], "-", color=c, lw=2, alpha=0.8)

            if show_std:
                for d in (r, w):
                    if "test_step_std" in d:
                        m = np.array(d["test_step"])
                        s = np.array(d["test_step_std"])
                        ax.fill_between(d["epochs"], m - s, m + s, color=c, alpha=0.15)

        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Test Step Accuracy", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_title(
            f"Accuracy for Learning a Dynamic Arbitration Policy (Part {chart_idx + 1}/{n_charts})",
            fontsize=14,
        )

        # Legend 1: Arbiter colors
        arbiter_handles = [
            mlines.Line2D(
                [],
                [],
                color=arbiter_colors[arb],
                linewidth=2,
                label=arb.replace("_", " ").title(),
            )
            for arb in group
        ]

        # Legend 2: Line styles
        style_handles = [
            mlines.Line2D(
                [], [], color="black", linewidth=2, linestyle="-", label="Warm Started"
            ),
            mlines.Line2D(
                [], [], color="black", linewidth=2, linestyle="--", label="Random Init"
            ),
        ]

        # Place both legends
        legend1 = ax.legend(
            handles=arbiter_handles, loc="lower right", title="Arbiter", fontsize=9
        )
        ax.add_artist(legend1)
        ax.legend(handles=style_handles, loc="upper left", title="Initialization")

        # Add trial count note
        fig.text(
            0.5,
            -0.02,
            f"Averaged over {n_random} control / {n_warm} warm-started trials",
            ha="center",
            fontsize=9,
            alpha=0.7,
        )

        plt.tight_layout()

        # Save each chart as separate file
        out_file = parent / f"{stem}_part{chart_idx + 1}{suffix}"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_file}")
        plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--random",
        nargs="+",
        required=True,
        help="Path(s) to JSON file(s) with random/control results",
    )
    parser.add_argument(
        "-w",
        "--warm",
        nargs="+",
        required=True,
        help="Path(s) to JSON file(s) with warm-started/test results",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output base path (e.g., comparison.png creates comparison_part1.png, etc.)",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=950,
        help="Maximum epoch to include (default: 950)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.9,
        help="Accuracy threshold for time-to-threshold analysis (default: 0.9)",
    )
    parser.add_argument(
        "--no-threshold-analysis",
        action="store_true",
        help="Disable threshold analysis output",
    )
    parser.add_argument(
        "--no-std", action="store_true", help="Disable standard deviation shading"
    )
    parser.add_argument(
        "--n-charts",
        type=int,
        default=3,
        help="Number of charts to split arbiters across (default: 3)",
    )

    args = parser.parse_args()

    print(f"Loading {len(args.random)} random/control file(s): {args.random}")
    random_data = load_and_average_trials(args.random, args.max_epoch)

    print(f"Loading {len(args.warm)} warm-started/test file(s): {args.warm}")
    warm_data = load_and_average_trials(args.warm, args.max_epoch)

    print(f"Found arbiters in random/control: {sorted(random_data.keys())}")
    print(f"Found arbiters in warm/test: {sorted(warm_data.keys())}")

    # Show what's missing from each
    only_in_random = set(random_data.keys()) - set(warm_data.keys())
    only_in_warm = set(warm_data.keys()) - set(random_data.keys())
    if only_in_random:
        print(f"Excluding (only in random/control): {sorted(only_in_random)}")
    if only_in_warm:
        print(f"Excluding (only in warm/test): {sorted(only_in_warm)}")

    if not args.no_threshold_analysis:
        # Filter to common arbiters for threshold analysis
        common = set(random_data.keys()) & set(warm_data.keys())
        random_common = {k: v for k, v in random_data.items() if k in common}
        warm_common = {k: v for k, v in warm_data.items() if k in common}

        analyze_random = compute_threshold_distribution(random_common, args.threshold)
        analyze_warm = compute_threshold_distribution(warm_common, args.threshold)

        random_times = summarize_distribution(
            analyze_random, "Random Init", args.threshold
        )
        warm_times = summarize_distribution(
            analyze_warm, "Warm Started", args.threshold
        )

        compare_distributions(
            random_times, warm_times, "Random Init", "Warm Started", args.threshold
        )

    plot_comparison(
        random_data,
        warm_data,
        args.output,
        show_std=not args.no_std,
        n_charts=args.n_charts,
    )


if __name__ == "__main__":
    main()
