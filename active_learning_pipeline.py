import glob
import os
import re
import subprocess
from datetime import datetime

import pandas as pd


def SynthesizeMealy(file_path):
    inputs = (
        subprocess.run(
            ["syfco", "--print-input-signals", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(" ", "")
        .strip()
    )

    outputs = (
        subprocess.run(
            ["syfco", "--print-output-signals", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(" ", "")
        .strip()
    )

    subprocess.run(
        f"ltlsynt --hide-status --tlsf {file_path} > System.hoa", shell=True, check=True
    )
    subprocess.run(
        "autfilt System.hoa --dot > System.dot",
        shell=True,
        check=True,
    )

    APs = subprocess.run(
        'grep "^AP:" System.hoa | sed "s/^AP: [0-9]* //" | tr -d \'"\' | tr " " ","',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return APs, inputs, outputs


def Active_Learning(Original_Dot_File, Inputs, Outputs):
    """Run active learning and capture the number of queries."""
    result = subprocess.run(
        f"python active_learning.py {Original_Dot_File} --inputs {Inputs} --outputs {Outputs}",
        check=True,
        shell=True,
        capture_output=True,
        text=True,
    )

    # Capture output for query count extraction
    output = result.stdout + result.stderr
    print(output)  # Print for visibility

    # Extract total membership queries from "Membership queries: X + Y = Z"
    num_queries = None
    match = re.search(r"Membership queries:.*=\s*(\d+)", output, re.IGNORECASE)
    if match:
        num_queries = int(match.group(1))

    subprocess.run(
        "autfilt System_learned.hoa --dot > System_learned.dot", shell=True, check=True
    )

    return num_queries, output


def Generate_And_Validate_Data(Learned_Dot_File, APs, Original_System):
    """Generate traces and validate, returning accuracy."""
    subprocess.run(
        f"python Dot_Trace_Generator.py {Learned_Dot_File} --fmt dot --aps {APs} -n 1000 -l 20 --cycle --out Active_Test_Dataset.txt",
        check=True,
        shell=True,
    )

    result = subprocess.run(
        f"python Trace_Checker.py {Original_System} Active_Test_Dataset.txt",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    print(output)  # Print for visibility

    # Extract acceptance percentage from "Acceptance %: X%"
    accuracy = None
    match = re.search(r"Acceptance\s*%?:\s*([0-9.]+)%?", output, re.IGNORECASE)
    if match:
        accuracy = float(match.group(1))

    return accuracy, output


def run_single_experiment(tlsf_path):
    """Run a single experiment and return (num_queries, accuracy)."""
    try:
        APs, inputs, outputs = SynthesizeMealy(tlsf_path)
        num_queries, learn_output = Active_Learning("System.dot", inputs, outputs)
        accuracy, validate_output = Generate_And_Validate_Data(
            "System_learned.dot", APs, "System.hoa"
        )
        return num_queries, accuracy, None
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {e.cmd}\nReturn code: {e.returncode}\nStderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}"
        print(f"Error: {error_msg}")
        return None, None, error_msg
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        return None, None, error_msg


def main(benchmark_dir, num_trials=3, output_csv="experiment_results.csv"):
    """Run experiments on all TLSF files in the directory."""

    # Find all TLSF files
    tlsf_files = glob.glob(os.path.join(benchmark_dir, "*.tlsf"))
    tlsf_files.sort()

    if not tlsf_files:
        print(f"No TLSF files found in {benchmark_dir}")
        return

    print(f"Found {len(tlsf_files)} TLSF files")
    print(f"Running {num_trials} trials per file")
    print("=" * 60)

    results = []

    for tlsf_path in tlsf_files:
        filename = os.path.basename(tlsf_path)
        print(f"\nProcessing: {filename}")
        print("-" * 40)

        for trial in range(1, num_trials + 1):
            print(f"  Trial {trial}/{num_trials}...")

            num_queries, accuracy, error = run_single_experiment(tlsf_path)

            result = {
                "tlsf_file": filename,
                "trial": trial,
                "sample_size": num_queries,
                "accuracy": accuracy,
            }
            results.append(result)

            if error:
                print(f"    ERROR: {error[:50]}...")
            else:
                print(f"    Sample Size: {num_queries}, Accuracy: {accuracy}")

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
        .agg({"sample_size": ["mean", "std"], "accuracy": ["mean", "std"]})
        .round(2)
    )
    summary.columns = [
        "avg_sample_size",
        "std_sample_size",
        "avg_accuracy",
        "std_accuracy",
    ]
    print(summary.to_string())

    return df


if __name__ == "__main__":
    benchmark_dir = "/workspaces/Automata_SSM_Learning/TestSet/SyntCompBenchMarks"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"active_learning_results_{timestamp}.csv"

    main(benchmark_dir, num_trials=3, output_csv=output_csv)
