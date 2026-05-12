#!/usr/bin/env python3
"""Passive Learning Experiment Runner.

Runs passive learning experiments on TLSF benchmarks, iteratively increasing
the number of traces until either:
  - Target accuracy is reached (default: 50%)
  - Maximum traces is reached (default: 100,000)

Starts at 5,000 traces and adds 5,000 each iteration.
"""

import glob
import os
import re
import subprocess
from datetime import datetime

import pandas as pd

# src/pipelines/ -> src/. Sibling scripts live in src/learning/, src/data/, src/scripts/.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_GEN = os.path.join(SRC_DIR, "data", "Dot_Trace_Generator.py")
TRACE_CHECKER = os.path.join(SRC_DIR, "scripts", "Trace_Checker.py")
PASSIVE_MEALY = os.path.join(SRC_DIR, "learning", "Passive_Mealy_Learning.py")


def SynthesizeMealy(file_path):
    # Get input/output signal NAMES from syfco (to know which are inputs vs outputs)
    input_names = set(
        subprocess.run(
            ["syfco", "--print-input-signals", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(" ", "")
        .strip()
        .split(",")
    )

    output_names = set(
        subprocess.run(
            ["syfco", "--print-output-signals", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(" ", "")
        .strip()
        .split(",")
    )

    try:
        subprocess.run(
            f"ltlsynt --hide-status --tlsf {file_path} > System.hoa",
            shell=True,
            check=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("[SKIP] synthesis timeout (>30s)")
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            raise RuntimeError("[SKIP] unrealizable")
        raise RuntimeError(f"[SKIP] ltlsynt failed (rc={e.returncode})")
    subprocess.run(
        "autfilt System.hoa --dot > System.dot",
        shell=True,
        check=True,
    )

    # Extract AP names FROM THE HOA FILE (preserves actual casing)
    ap_line = subprocess.run(
        'grep "^AP:" System.hoa',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Parse: AP: 5 "finished_0" "finished_1" ... -> list of names
    ap_names = re.findall(r'"([^"]+)"', ap_line)

    # Match against input/output names (case-insensitive)
    input_names_lower = {n.lower() for n in input_names}
    output_names_lower = {n.lower() for n in output_names}

    actual_inputs = [ap for ap in ap_names if ap.lower() in input_names_lower]
    actual_outputs = [ap for ap in ap_names if ap.lower() in output_names_lower]

    inputs = ",".join(actual_inputs)
    outputs = ",".join(actual_outputs)
    APs = ",".join(ap_names)

    return APs, inputs, outputs


def GenerateTraces(dot_file, aps, num_traces, trace_length, output_file):
    """Generate traces from a DOT file."""
    subprocess.run(
        [
            "python",
            TRACE_GEN,
            dot_file,
            "--fmt",
            "dot",
            "--aps",
            aps,
            "-n",
            str(num_traces),
            "-l",
            str(trace_length),
            "--cycle",
            "--out",
            output_file,
        ],
        check=True,
    )


def CheckTraces(hoa_file, data_file):
    """Run Trace_Checker and parse acceptance percentage from stdout."""
    result = subprocess.run(
        ["python", TRACE_CHECKER, hoa_file, data_file],
        capture_output=True,
        text=True,
        check=True,
    )

    acc = None
    for line in result.stdout.splitlines():
        if "Acceptance" in line and "%" in line:
            # Handle formats like "Acceptance %: 95.5%" or "Acceptance: 95.5%"
            match = re.search(r"([0-9.]+)%?", line.split(":")[-1])
            if match:
                acc = float(match.group(1))
    return acc


def PassiveLearning(data_file, inputs, outputs):
    """Run passive Mealy learning on a trace file."""
    subprocess.run(
        ["python", PASSIVE_MEALY, data_file, inputs, outputs],
        check=True,
    )

    subprocess.run(
        "autfilt Training_Dataset.hoa --dot > Training_Dataset.dot",
        shell=True,
        check=True,
    )


def PassiveLearningWithConvergence(
    tlsf_file,
    start_traces=5000,
    step_traces=5000,
    max_traces=50000,
    target_accuracy=100.0,
    trace_length=20,
    test_traces=1000,
):
    """Iteratively run passive learning, increasing trace count until accuracy >=
    target_accuracy or num_traces >= max_traces.

    Args:
        tlsf_file: Path to the TLSF specification file
        start_traces: Initial number of traces to generate
        step_traces: Number of traces to add each iteration
        max_traces: Maximum number of traces before stopping
        target_accuracy: Target accuracy percentage to achieve
        trace_length: Length of each trace
        test_traces: Number of traces to use for testing

    Returns:
        tuple: (final_num_traces, final_accuracy, history)
            - final_num_traces: Number of traces when stopped
            - final_accuracy: Accuracy achieved when stopped
            - history: List of (num_traces, accuracy) tuples for each iteration
    """
    APs, Inputs, Outputs = SynthesizeMealy(tlsf_file)

    history = []
    num_traces = start_traces

    while num_traces <= max_traces:
        print(f"    Trying {num_traces} traces...")

        # Generate training data
        GenerateTraces(
            "System.dot", APs, num_traces, trace_length, "Training_Dataset.txt"
        )

        # Learn from traces
        PassiveLearning("Training_Dataset.txt", Inputs, Outputs)

        # Generate test data from learned model and check accuracy
        GenerateTraces(
            "Training_Dataset.dot", APs, test_traces, trace_length, "Test_Dataset.txt"
        )
        acc = CheckTraces("System.hoa", "Test_Dataset.txt")

        history.append((num_traces, acc))
        print(f"      Accuracy: {acc}%")

        # Check stopping conditions
        if acc is not None and acc >= target_accuracy:
            print(
                f"    ✓ Reached target accuracy {target_accuracy}% with {num_traces} traces"
            )
            return num_traces, acc, history

        num_traces += step_traces

    # Hit max traces without reaching target
    final_traces, final_acc = history[-1] if history else (None, None)
    print(f"    ✗ Reached max traces ({max_traces}) with accuracy {final_acc}%")
    return final_traces, final_acc, history


def run_single_experiment(
    tlsf_path,
    start_traces=5000,
    step_traces=5000,
    max_traces=100000,
    target_accuracy=50.0,
    trace_length=20,
    test_traces=1000,
):
    """Run a single passive learning experiment with convergence."""
    try:
        final_traces, final_acc, history = PassiveLearningWithConvergence(
            tlsf_path,
            start_traces=start_traces,
            step_traces=step_traces,
            max_traces=max_traces,
            target_accuracy=target_accuracy,
            trace_length=trace_length,
            test_traces=test_traces,
        )

        return {
            "num_traces": final_traces,
            "sample_size": final_traces,
            "accuracy": final_acc,
            "converged": (
                final_acc >= target_accuracy if final_acc is not None else False
            ),
            "history": history,
            "error": None,
        }

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {e.cmd}\nReturn code: {e.returncode}"
        if hasattr(e, "stderr") and e.stderr:
            error_msg += f"\nStderr: {e.stderr}"
        print(f"    ERROR: {error_msg}")
        return {
            "num_traces": None,
            "accuracy": None,
            "sample_size": None,
            "converged": False,
            "history": [],
            "error": error_msg,
        }

    except Exception as e:
        error_msg = str(e)
        if error_msg.startswith("[SKIP]"):
            print(f"    {error_msg}")
        else:
            print(f"    ERROR: {error_msg}")
        return {
            "num_traces": None,
            "accuracy": None,
            "sample_size": None,
            "converged": False,
            "history": [],
            "error": error_msg,
        }


def main(
    benchmark_dir,
    num_trials=3,
    output_csv="passive_learning_results.csv",
    start_traces=5000,
    step_traces=5000,
    max_traces=100000,
    target_accuracy=50.0,
    trace_length=20,
    test_traces=1000,
):
    """Run passive learning experiments on all TLSF files in the directory.

    Args:
        benchmark_dir: Directory containing TLSF files
        num_trials: Number of trials per file
        output_csv: Output CSV file path
        start_traces: Initial number of traces
        step_traces: Traces to add each iteration
        max_traces: Maximum traces before stopping
        target_accuracy: Target accuracy to achieve
        trace_length: Length of each trace
        test_traces: Number of test traces for validation
    """
    # Find all TLSF files
    tlsf_files = glob.glob(os.path.join(benchmark_dir, "*.tlsf"))
    tlsf_files.sort()

    if not tlsf_files:
        print(f"No TLSF files found in {benchmark_dir}")
        return None

    print(f"Found {len(tlsf_files)} TLSF files")
    print(f"Running {num_trials} trials per file")
    print("Settings:")
    print(f"  Start traces: {start_traces}")
    print(f"  Step traces: {step_traces}")
    print(f"  Max traces: {max_traces}")
    print(f"  Target accuracy: {target_accuracy}%")
    print(f"  Trace length: {trace_length}")
    print("=" * 60)

    results = []

    for tlsf_path in tlsf_files:
        filename = os.path.basename(tlsf_path)
        print(f"\nProcessing: {filename}")
        print("-" * 40)

        for trial in range(1, num_trials + 1):
            print(f"  Trial {trial}/{num_trials}...")

            result = run_single_experiment(
                tlsf_path,
                start_traces=start_traces,
                step_traces=step_traces,
                max_traces=max_traces,
                target_accuracy=target_accuracy,
                trace_length=trace_length,
                test_traces=test_traces,
            )

            row = {
                "tlsf_file": filename,
                "trial": trial,
                "num_traces": result["num_traces"],
                "sample_size": result["sample_size"],
                "accuracy": result["accuracy"],
                "converged": result["converged"],
            }
            results.append(row)

            if result["error"]:
                print(f"    FAILED: {result['error'][:50]}...")
            else:
                print(
                    f"    Result: {result['num_traces']} traces, "
                    f"{result['accuracy']}% accuracy, "
                    f"converged={result['converged']}"
                )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_csv}")

    # Print the full DataFrame
    print("\n=== RESULTS DATAFRAME ===")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.to_string(index=False))

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    summary = (
        df.groupby("tlsf_file")
        .agg(
            {
                "num_traces": ["mean", "std"],
                "sample_size": ["mean", "std"],
                "accuracy": ["mean", "std"],
                "converged": ["sum", "count"],
            }
        )
        .round(2)
    )
    summary.columns = [
        "avg_traces",
        "std_traces",
        "avg_sample_size",
        "std_sample_size",
        "avg_accuracy",
        "std_accuracy",
        "num_converged",
        "total_trials",
    ]
    print(summary.to_string())

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run passive learning experiments on TLSF files"
    )
    parser.add_argument(
        "--benchmark-dir",
        "-d",
        default="symbolic_vs_Gradient_learning_TLSF",
        help="Directory containing TLSF files (non-recursive)",
    )
    parser.add_argument(
        "--trials", "-t", type=int, default=3, help="Number of trials per file"
    )
    parser.add_argument("--output", "-o", help="Output CSV file (default: timestamped)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output or f"passive_learning_results_{timestamp}.csv"

    main(
        args.benchmark_dir,
        num_trials=args.trials,
        output_csv=output_csv,
        start_traces=5000,
        step_traces=5000,
        max_traces=50000,
        target_accuracy=100.0,
        trace_length=20,
        test_traces=1000,
    )
