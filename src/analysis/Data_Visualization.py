#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------


def normalize_tlsf(path):
    """Normalize TLSF paths so all datasets join cleanly."""
    if path is None:
        return None
    return path.split("TestSet/")[-1]


# -----------------------------
# Loaders
# -----------------------------


def load_max_epoch_df(json_dir):
    rows = []
    for json_path in Path(json_dir).glob("*.json"):
        with open(json_path, "r") as f:
            data = json.load(f)

        epoch_hist = data.get("epoch_history", [])
        max_entry = (
            max(epoch_hist, key=lambda x: x.get("epoch", -1)) if epoch_hist else {}
        )

        rows.append(
            {
                "json_file": json_path.name,
                "tlsf_file": data.get("file"),
                "error": data.get("error"),
                "success": data.get("error") is None,
                # run-level metadata
                "training_samples": data.get("training_samples"),
                "converged_epoch": data.get("converged_epoch"),
                "sample_complexity": data.get("sample_complexity"),
                "acceptance": data.get("acceptance"),
                # max-epoch metrics
                "epoch": max_entry.get("epoch"),
                "loss": max_entry.get("loss"),
                "test_step_acc": max_entry.get("test_step_acc"),
                "test_trace_acc": max_entry.get("test_trace_acc"),
                "num_epochs_logged": len(epoch_hist),
            }
        )

    return pd.DataFrame(rows)


def load_active_learning_results(json_dir):
    rows = []
    for json_path in Path(json_dir).glob("*.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            rows.append(
                {
                    "json_file": json_path.name,
                    "tlsf_file": f'/workspaces/Automata_SSM_Learning/TestSet/{data.get("tlsf_file")}',
                    "trial": data.get("trial"),
                    "sample_size": data.get("sample_size"),
                    "accuracy": data.get("accuracy"),
                    "status": data.get("status"),
                    "success": data.get("status") == "success",
                }
            )
        except Exception:
            rows.append(
                {
                    "json_file": json_path.name,
                    "tlsf_file": None,
                    "trial": None,
                    "sample_size": None,
                    "accuracy": None,
                    "status": "parse_error",
                    "success": False,
                }
            )
    return pd.DataFrame(rows)


def load_passive_learning_results(json_dir):
    rows = []
    for json_path in Path(json_dir).glob("*.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            results = data.get("results", [])
            if results:
                max_entry = max(results, key=lambda x: x.get("accuracy", float("-inf")))
                max_accuracy = max_entry.get("accuracy")
                max_num_traces = max_entry.get("num_traces")
            else:
                max_accuracy = None
                max_num_traces = None

            rows.append(
                {
                    "json_file": json_path.name,
                    "tlsf_file": f'/workspaces/Automata_SSM_Learning/TestSet/{data.get("tlsf_file")}',
                    "trial": data.get("trial"),
                    "trace_length": data.get("trace_length"),
                    "max_accuracy": max_accuracy,
                    "num_traces_at_max": max_num_traces,
                    "status": data.get("status"),
                    "success": data.get("status") == "success",
                }
            )
        except Exception:
            rows.append(
                {
                    "json_file": json_path.name,
                    "tlsf_file": None,
                    "trial": None,
                    "trace_length": None,
                    "max_accuracy": None,
                    "num_traces_at_max": None,
                    "status": "parse_error",
                    "success": False,
                }
            )
    return pd.DataFrame(rows)


def load_synth_stats(json_path):
    """Load ltlsynt statistics from tlsf_synth_results.json."""
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        tlsf_file = entry.get("tlsf_file", "")
        # Normalize to match other datasets
        normalized = normalize_tlsf(tlsf_file)

        num_inputs = entry.get("num_inputs") or 0
        num_outputs = entry.get("num_outputs") or 0

        rows.append(
            {
                "tlsf_file": normalized,
                "num_states": entry.get("num_states"),
                "num_inputs": num_inputs,
                "num_outputs": num_outputs,
                "num_aps": num_inputs + num_outputs,
                "realizable": entry.get("realizable"),
                "synth_success": entry.get("success"),
            }
        )

    return pd.DataFrame(rows)


def make_benchmark_plot(df, output_prefix):
    """Generate accuracy + sample complexity plots sorted by SSM accuracy descending."""

    # Sort by SSM accuracy descending
    plot_df = df.sort_values("ssm_max_accuracy", ascending=False).reset_index(drop=True)

    # Shorten TLSF file names for display
    plot_df["tlsf_short"] = plot_df["tlsf_file"].apply(
        lambda x: Path(x).stem if x else x
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(plot_df) * 0.7), 10), sharex=True
    )

    x = range(len(plot_df))
    width = 0.25

    # Top plot: Accuracy
    ax1.scatter(
        [i - width for i in x],
        plot_df["ssm_max_accuracy"],
        label="SSM",
        marker="o",
        s=80,
        color="tab:blue",
        zorder=3,
    )
    ax1.scatter(
        [i for i in x],
        plot_df["active_accuracy"],
        label="Active Learning (L*)",
        marker="s",
        s=80,
        color="tab:orange",
        zorder=3,
    )
    ax1.scatter(
        [i + width for i in x],
        plot_df["passive_accuracy"],
        label="Passive Learning (RPNI)",
        marker="^",
        s=80,
        color="tab:green",
        zorder=3,
    )

    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")
    ax1.set_title(
        "Accuracy Comparison by TLSF Benchmark (sorted by SSM accuracy)", fontsize=14
    )
    ax1.grid(axis="y", alpha=0.3)

    # Bottom plot: Sample Size (log scale)
    ssm_samples = plot_df["ssm_samples"].replace(0, 1)
    active_samples = plot_df["active_samples"].replace(0, 1)
    passive_samples = plot_df["passive_samples"].replace(0, 1)

    ax2.scatter(
        [i - width for i in x],
        ssm_samples,
        label="SSM",
        marker="o",
        s=80,
        color="tab:blue",
        zorder=3,
    )
    ax2.scatter(
        [i for i in x],
        active_samples,
        label="Active Learning (L*)",
        marker="s",
        s=80,
        color="tab:orange",
        zorder=3,
    )
    ax2.scatter(
        [i + width for i in x],
        passive_samples,
        label="Passive Learning (RPNI)",
        marker="^",
        s=80,
        color="tab:green",
        zorder=3,
    )

    ax2.set_ylabel("Sample Size (log scale)", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(loc="upper left")
    ax2.set_title("Sample Complexity by TLSF Benchmark", fontsize=14)
    ax2.grid(axis="y", alpha=0.3)

    # X-axis labels
    plt.xticks(x, plot_df["tlsf_short"], rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("TLSF Benchmark File", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_prefix}.png / .pdf ({len(plot_df)} benchmarks)")


def make_grouped_line_chart(df, group_col, group_label, output_prefix, min_count=2):
    """Generate line chart showing mean accuracy per group for each method.

    Filters out groups with fewer than min_count benchmarks to remove outliers.
    """

    # Group by the column and compute mean accuracy
    grouped = (
        df.groupby(group_col, as_index=False)
        .agg(
            ssm_accuracy=("ssm_max_accuracy", "mean"),
            active_accuracy=("active_accuracy", "mean"),
            passive_accuracy=("passive_accuracy", "mean"),
            count=("tlsf_file", "count"),
        )
        .sort_values(group_col)
    )

    # Filter out groups with too few samples (outliers)
    grouped = grouped[grouped["count"] >= min_count]

    if len(grouped) == 0:
        print(f"Warning: No groups with >= {min_count} samples for {group_label}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = grouped[group_col].values

    ax.plot(
        x,
        grouped["ssm_accuracy"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="SSM",
        color="tab:blue",
    )
    ax.plot(
        x,
        grouped["active_accuracy"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="Active Learning (L*)",
        color="tab:orange",
    )
    ax.plot(
        x,
        grouped["passive_accuracy"],
        marker="^",
        linewidth=2,
        markersize=8,
        label="Passive Learning (RPNI)",
        color="tab:green",
    )

    ax.set_xlabel(group_label, fontsize=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    ax.set_title(
        f"Mean Accuracy by {group_label} (n ≥ {min_count} per group)", fontsize=14
    )
    ax.grid(alpha=0.3)

    # Add count annotations
    for xi, cnt in zip(x, grouped["count"]):
        ax.annotate(f"n={cnt}", (xi, 0.02), ha="center", fontsize=8, alpha=0.7)

    # Set x-ticks to only show integer values present in data
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_prefix}.png / .pdf (groups with n >= {min_count})")


def export_latex_table(df, output_path):
    """Export the combined results as a LaTeX table."""

    # Select and rename columns for the table
    table_df = df[
        [
            "tlsf_file",
            "ssm_max_accuracy",
            "active_accuracy",
            "passive_accuracy",
            "ssm_samples",
            "active_samples",
            "passive_samples",
        ]
    ].copy()

    # Shorten TLSF file names and wrap in mathtt with line breaking
    def format_benchmark_name(x):
        if not x:
            return x
        stem = Path(x).stem.replace("_", "\\_")

        # Break long names into chunks of ~16 chars at hyphens or underscores
        if len(stem) > 16:
            parts = []
            current = ""
            for char in stem:
                current += char
                if char == "-" and len(current) >= 16:
                    parts.append(current)
                    current = ""
            if current:
                parts.append(current)
            # Each part gets its own $\mathtt{}$
            formatted_parts = [f"$\\mathtt{{{p}}}$" for p in parts]
            separator = " \\\\ "
            joined = separator.join(formatted_parts)
            return f"\\makecell[l]{{{joined}}}"

        return f"$\\mathtt{{{stem}}}$"

    table_df["tlsf_file"] = table_df["tlsf_file"].apply(format_benchmark_name)

    # Sort by TLSF file name
    table_df = table_df.sort_values("tlsf_file").reset_index(drop=True)

    # Format accuracies as percentages
    table_df["ssm_max_accuracy"] = table_df["ssm_max_accuracy"].apply(
        lambda x: f"{x*100:.1f}"
    )
    table_df["active_accuracy"] = table_df["active_accuracy"].apply(
        lambda x: f"{x*100:.1f}"
    )
    table_df["passive_accuracy"] = table_df["passive_accuracy"].apply(
        lambda x: f"{x*100:.1f}"
    )

    # Format sample counts as integers
    table_df["ssm_samples"] = table_df["ssm_samples"].apply(lambda x: f"{int(x):,}")
    table_df["active_samples"] = table_df["active_samples"].apply(
        lambda x: f"{int(x):,}" if x > 0 else "—"
    )
    table_df["passive_samples"] = table_df["passive_samples"].apply(
        lambda x: f"{int(x):,}"
    )

    # Rename columns for LaTeX
    table_df.columns = [
        "Benchmark",
        "SSM Acc",
        "L* Acc",
        "RPNI Acc",
        "SSM Samples",
        "L* Samples",
        "RPNI Samples",
    ]

    # Generate LaTeX using longtable for multi-page support
    latex_lines = []
    latex_lines.append(
        "\\begin{small}  % Options: \\tiny, \\scriptsize, \\footnotesize, \\small, \\normalsize"
    )
    latex_lines.append("\\begin{longtable}{p{5cm}|rrr|rrr}")
    latex_lines.append(
        "\\caption{Comparison of learning methods on SYNTCOMP benchmarks.} \\label{tab:benchmark_comparison} \\\\"
    )
    latex_lines.append("\\toprule")
    latex_lines.append(
        "Benchmark & SSM Acc (\\%) & L* Acc (\\%) & RPNI Acc (\\%) & SSM Samples & L* Samples & RPNI Samples \\\\"
    )
    latex_lines.append("\\midrule")
    latex_lines.append("\\endfirsthead")
    latex_lines.append("")
    latex_lines.append(
        "\\multicolumn{7}{c}{\\tablename\\ \\thetable{} -- continued from previous page} \\\\"
    )
    latex_lines.append("\\toprule")
    latex_lines.append(
        "Benchmark & SSM Acc (\\%) & L* Acc (\\%) & RPNI Acc (\\%) & SSM Samples & L* Samples & RPNI Samples \\\\"
    )
    latex_lines.append("\\midrule")
    latex_lines.append("\\endhead")
    latex_lines.append("")
    latex_lines.append("\\midrule")
    latex_lines.append("\\multicolumn{7}{r}{Continued on next page} \\\\")
    latex_lines.append("\\endfoot")
    latex_lines.append("")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\endlastfoot")
    latex_lines.append("")

    # Add data rows
    for _, row in table_df.iterrows():
        latex_lines.append(
            f"{row['Benchmark']} & {row['SSM Acc']} & {row['L* Acc']} & {row['RPNI Acc']} & {row['SSM Samples']} & {row['L* Samples']} & {row['RPNI Samples']} \\\\"
        )

    latex_lines.append("\\end{longtable}")
    latex_lines.append("\\end{small}")

    latex_output = "\n".join(latex_lines)

    with open(output_path, "w") as f:
        f.write("% Requires: \\usepackage{longtable, booktabs, makecell}\n")
        f.write(latex_output)

    print(f"Saved LaTeX table to {output_path}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze SSM, Active, and Passive learning results"
    )
    parser.add_argument(
        "--latex", action="store_true", help="Export full results as LaTeX table"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for plots and tables",
    )
    args = parser.parse_args()

    # Load learning results
    SSM_DF = load_max_epoch_df(
        "/workspaces/Automata_SSM_Learning/SSM_Syntcomp_Benchmarks"
    )
    ACTIVE_DF = load_active_learning_results(
        "/workspaces/Automata_SSM_Learning/lambda/results-active"
    )
    PASSIVE_DF = load_passive_learning_results(
        "/workspaces/Automata_SSM_Learning/lambda/results-passive"
    )

    # Load synthesis stats
    SYNTH_DF = load_synth_stats(
        "/workspaces/Automata_SSM_Learning/tlsf_synth_results.json"
    )

    # Normalize TLSF paths
    SSM_DF["tlsf_file"] = SSM_DF["tlsf_file"].apply(normalize_tlsf)
    ACTIVE_DF["tlsf_file"] = ACTIVE_DF["tlsf_file"].apply(normalize_tlsf)
    PASSIVE_DF["tlsf_file"] = PASSIVE_DF["tlsf_file"].apply(normalize_tlsf)

    # -----------------------------
    # Aggregate
    # -----------------------------

    # SSM: max accuracy, samples = 9000 * epoch (earliest max)
    # Filter out any with 0 samples (invalid)
    SSM_AGG = (
        SSM_DF.dropna(subset=["epoch", "test_trace_acc"])
        .assign(ssm_samples=lambda df: 9000 * df["epoch"])
        .query("ssm_samples > 0")
        .groupby("tlsf_file", as_index=False)
        .agg(
            ssm_max_accuracy=("test_trace_acc", "max"),
            ssm_samples=("ssm_samples", "min"),
        )
    )

    # Active: include all benchmarks, fill failures with 0
    ACTIVE_AGG = ACTIVE_DF.groupby("tlsf_file", as_index=False).agg(
        active_accuracy=("accuracy", lambda x: x.mean() if x.notna().any() else 0.0),
        active_samples=("sample_size", lambda x: x.mean() if x.notna().any() else 0.0),
        num_trials=("trial", "nunique"),
    )
    # scale accuracy to [0,1], handle NaN -> 0
    ACTIVE_AGG["active_accuracy"] = ACTIVE_AGG["active_accuracy"].fillna(0.0) / 100.0
    ACTIVE_AGG["active_samples"] = ACTIVE_AGG["active_samples"].fillna(0.0)

    # Passive: max accuracy, samples = num_traces * trace_length / 20 (must succeed)
    PASSIVE_AGG = (
        PASSIVE_DF.dropna(subset=["max_accuracy", "num_traces_at_max", "trace_length"])
        .assign(
            passive_samples=lambda df: df["num_traces_at_max"] * df["trace_length"] / 20
        )
        .groupby("tlsf_file", as_index=False)
        .agg(
            passive_accuracy=("max_accuracy", "max"),
            passive_samples=("passive_samples", "min"),
        )
    )
    PASSIVE_AGG["passive_accuracy"] /= 100.0

    # -----------------------------
    # Join
    # -----------------------------

    # Inner join on SSM and Passive (must succeed), left join Active (failures -> 0)
    COMBINED = (
        SSM_AGG.merge(PASSIVE_AGG, on="tlsf_file", how="inner")
        .merge(ACTIVE_AGG, on="tlsf_file", how="left")
        .merge(SYNTH_DF, on="tlsf_file", how="left")
    )
    # Fill any missing active results with 0
    COMBINED["active_accuracy"] = COMBINED["active_accuracy"].fillna(0.0)
    COMBINED["active_samples"] = COMBINED["active_samples"].fillna(0.0)

    print(f"Combined dataset: {len(COMBINED)} benchmarks")
    print(f"  With synth stats: {COMBINED['num_states'].notna().sum()}")

    # Calculate percentage of benchmarks solved at 100% accuracy
    ssm_perfect = (COMBINED["ssm_max_accuracy"] >= 0.999).sum()
    active_perfect = (COMBINED["active_accuracy"] >= 0.999).sum()
    passive_perfect = (COMBINED["passive_accuracy"] >= 0.999).sum()
    total = len(COMBINED)

    print("\nPerfect accuracy (100%) statistics:")
    print(f"  SSM:     {ssm_perfect}/{total} ({100*ssm_perfect/total:.1f}%)")
    print(f"  L*:      {active_perfect}/{total} ({100*active_perfect/total:.1f}%)")
    print(f"  RPNI:    {passive_perfect}/{total} ({100*passive_perfect/total:.1f}%)")

    if len(COMBINED) == 0:
        print("No overlapping benchmarks found between all three datasets!")
        print(f"SSM benchmarks: {len(SSM_AGG)}")
        print(f"Active benchmarks: {len(ACTIVE_AGG)}")
        print(f"Passive benchmarks: {len(PASSIVE_AGG)}")
        exit(1)

    # -----------------------------
    # LaTeX export (full dataset)
    # -----------------------------

    if args.latex:
        export_latex_table(COMBINED, f"{args.output_dir}/benchmark_results.tex")
        print(f"\nExported LaTeX table for {len(COMBINED)} benchmarks")

    # -----------------------------
    # Generate plots
    # -----------------------------

    # Only include rows with valid synth stats for grouped plots
    COMBINED_WITH_SYNTH = COMBINED.dropna(
        subset=["num_states", "num_inputs", "num_outputs"]
    )

    # Random sample of 20 benchmarks
    SAMPLE_SIZE = 20
    if len(COMBINED_WITH_SYNTH) > SAMPLE_SIZE:
        COMBINED_WITH_SYNTH = COMBINED_WITH_SYNTH.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"\nRandomly sampled {SAMPLE_SIZE} benchmarks from {len(COMBINED)} total")

    print(
        f"\nGenerating plots for {len(COMBINED_WITH_SYNTH)} benchmarks with synth stats..."
    )

    # 1. Main benchmark plot sorted by SSM accuracy descending
    make_benchmark_plot(COMBINED_WITH_SYNTH, f"{args.output_dir}/benchmark_comparison")

    # 2. Grouped line chart by number of APs
    make_grouped_line_chart(
        COMBINED_WITH_SYNTH,
        "num_aps",
        "Number of APs",
        f"{args.output_dir}/accuracy_by_aps",
    )

    # 3. Grouped line chart by number of states
    make_grouped_line_chart(
        COMBINED_WITH_SYNTH,
        "num_states",
        "Number of States",
        f"{args.output_dir}/accuracy_by_states",
    )

    # 4. Grouped line chart by number of inputs
    make_grouped_line_chart(
        COMBINED_WITH_SYNTH,
        "num_inputs",
        "Number of Inputs",
        f"{args.output_dir}/accuracy_by_inputs",
    )

    # 5. Grouped line chart by number of outputs
    make_grouped_line_chart(
        COMBINED_WITH_SYNTH,
        "num_outputs",
        "Number of Outputs",
        f"{args.output_dir}/accuracy_by_outputs",
    )

    # -----------------------------
    # Print summary table
    # -----------------------------

    print("\nSummary table:")
    summary_cols = [
        "tlsf_file",
        "num_states",
        "num_inputs",
        "num_outputs",
        "num_aps",
        "ssm_max_accuracy",
        "active_accuracy",
        "passive_accuracy",
        "ssm_samples",
        "active_samples",
        "passive_samples",
    ]
    print(COMBINED_WITH_SYNTH[summary_cols].to_string())

    # Save combined data to CSV
    COMBINED.to_csv(f"{args.output_dir}/benchmark_combined_results.csv", index=False)
    print(
        f"\nSaved combined results to {args.output_dir}/benchmark_combined_results.csv"
    )
