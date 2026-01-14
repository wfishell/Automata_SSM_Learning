import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from multiprocessing import Pool

import pandas as pd

# Timeout in seconds for active learning
ACTIVE_LEARNING_TIMEOUT = 30  # 2 minutes
NUM_WORKERS = 16  # Number of parallel workers

# Store the directory where the script is run from (where active_learning.py etc. live)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Active learning timed out")


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

    subprocess.run(
        f"ltlsynt --hide-status --tlsf {file_path} > System.hoa", shell=True, check=True
    )
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


def Active_Learning(
    Original_Dot_File, Inputs, Outputs, timeout_seconds=ACTIVE_LEARNING_TIMEOUT
):
    """Run active learning and capture the number of queries with timeout."""
    try:
        # Use absolute path to active_learning.py
        active_learning_script = os.path.join(SCRIPT_DIR, "active_learning.py")

        result = subprocess.run(
            f"python {active_learning_script} {Original_Dot_File} --inputs {Inputs} --outputs {Outputs}",
            check=True,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
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
            "autfilt System_learned.hoa --dot > System_learned.dot",
            shell=True,
            check=True,
        )
        return num_queries, output

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Active learning timed out after {timeout_seconds} seconds")


def Generate_And_Validate_Data(Learned_Dot_File, APs, Original_System):
    """Generate traces and validate, returning accuracy."""
    # Use absolute paths to scripts
    trace_gen_script = os.path.join(SCRIPT_DIR, "Dot_Trace_Generator.py")
    trace_check_script = os.path.join(SCRIPT_DIR, "Trace_Checker.py")

    subprocess.run(
        f"python {trace_gen_script} {Learned_Dot_File} --fmt dot --aps {APs} -n 1000 -l 20 --cycle --out Active_Test_Dataset.txt",
        check=True,
        shell=True,
    )

    result = subprocess.run(
        f"python {trace_check_script} {Original_System} Active_Test_Dataset.txt",
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
    """Run a single experiment in an isolated temp directory."""
    # Each worker gets its own temp directory to avoid file conflicts
    original_dir = os.getcwd()
    work_dir = tempfile.mkdtemp(prefix="active_learn_")

    try:
        os.chdir(work_dir)

        APs, inputs, outputs = SynthesizeMealy(tlsf_path)
        num_queries, learn_output = Active_Learning("System.dot", inputs, outputs)
        accuracy, validate_output = Generate_And_Validate_Data(
            "System_learned.dot", APs, "System.hoa"
        )
        return num_queries, accuracy, None

    except TimeoutError as e:
        error_msg = f"TIMEOUT: {str(e)}"
        print(f"Error: {error_msg}")
        return None, 0, error_msg

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {e.cmd}\nReturn code: {e.returncode}\nStderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}"
        print(f"Error: {error_msg}")
        return None, None, error_msg

    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        return None, None, error_msg

    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


def run_experiment_worker(task):
    """Worker function for parallel processing.

    Takes a task tuple: (tlsf_path, tlsf_name, trial)
    Returns a result dict.
    """
    tlsf_path, tlsf_name, trial = task

    print(f"[Worker {os.getpid()}] Starting: {tlsf_name} (trial {trial})")

    num_queries, accuracy, error = run_single_experiment(tlsf_path)

    result = {
        "tlsf_file": tlsf_name,
        "trial": trial,
        "sample_size": num_queries,
        "accuracy": accuracy,
    }

    if error:
        if "TIMEOUT" in error:
            print(f"[Worker {os.getpid()}] {tlsf_name} trial {trial}: TIMEOUT")
        else:
            print(
                f"[Worker {os.getpid()}] {tlsf_name} trial {trial}: ERROR - {error[:40]}..."
            )
    else:
        print(
            f"[Worker {os.getpid()}] {tlsf_name} trial {trial}: queries={num_queries}, acc={accuracy}"
        )

    return result


def find_tlsf_files_recursive(benchmark_dir):
    """Recursively find all TLSF files and return list of (full_path, relative_name)."""
    tlsf_files = []

    for root, dirs, files in os.walk(benchmark_dir):
        for file in files:
            if file.endswith(".tlsf"):
                full_path = os.path.join(root, file)
                # Get relative path from benchmark_dir
                rel_path = os.path.relpath(full_path, benchmark_dir)
                # Convert path separators to underscores for the name
                # e.g., "amba/amba_02.tlsf" -> "amba_amba_02.tlsf"
                rel_name = rel_path.replace(os.sep, "_")
                tlsf_files.append((full_path, rel_name))

    tlsf_files.sort(key=lambda x: x[1])
    return tlsf_files


def main(
    benchmark_dir,
    num_trials=3,
    output_csv="experiment_results.csv",
    parallel=True,
    num_workers=NUM_WORKERS,
):
    """Run experiments on all TLSF files in the directory and subdirectories.

    How parallelization works:
    1. Find all TLSF files recursively in benchmark_dir
    2. Create a list of tasks: [(file1, name1, trial1), (file1, name1, trial2), ...]
    3. Create a Pool of N worker processes
    4. Pool.map() distributes tasks across workers automatically
       - Each worker picks up the next available task when it finishes
       - No redundant runs: each (file, trial) pair is processed exactly once
       - Load balancing: fast tasks don't block slow ones

    Args:
        benchmark_dir: Root directory to search for .tlsf files
        num_trials: Number of trials per file
        output_csv: Output CSV filename
        parallel: If True, use multiprocessing; if False, run serially
        num_workers: Number of parallel worker processes
    """

    # Find all TLSF files recursively
    tlsf_files = find_tlsf_files_recursive(benchmark_dir)

    if not tlsf_files:
        print(f"No TLSF files found in {benchmark_dir}")
        return

    print(f"Found {len(tlsf_files)} TLSF files")
    print(f"Running {num_trials} trial(s) per file")
    print(f"Total tasks: {len(tlsf_files) * num_trials}")
    print(f"Active learning timeout: {ACTIVE_LEARNING_TIMEOUT} seconds")
    print(f"Parallel: {parallel}, Workers: {num_workers if parallel else 1}")
    print("=" * 60)

    # Build list of all tasks: (tlsf_path, tlsf_name, trial)
    # Each task is a unique (file, trial) combination - no redundancy
    tasks = []
    for tlsf_path, tlsf_name in tlsf_files:
        for trial in range(1, num_trials + 1):
            tasks.append((tlsf_path, tlsf_name, trial))

    print("\nTask distribution:")
    print(f"  - {len(tlsf_files)} files × {num_trials} trials = {len(tasks)} tasks")
    print(f"  - {num_workers} workers will process tasks in parallel")
    print("  - Each task runs in isolated temp directory (no file conflicts)")
    print("=" * 60)

    if parallel and num_workers > 1:
        # Run in parallel using process pool
        # Pool.map ensures each task is processed exactly once
        with Pool(processes=num_workers) as pool:
            results = pool.map(run_experiment_worker, tasks)
    else:
        # Run in series (for debugging or single-core systems)
        results = [run_experiment_worker(task) for task in tasks]

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Run active learning experiments on TLSF files"
    )
    parser.add_argument(
        "--benchmark-dir",
        "-d",
        default="/workspaces/Automata_SSM_Learning/TestSet/benchmarks/tlsf",
        help="Directory containing TLSF files (searched recursively)",
    )
    parser.add_argument(
        "--trials", "-t", type=int, default=1, help="Number of trials per file"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=NUM_WORKERS,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--serial", action="store_true", help="Run in serial mode (no parallelization)"
    )
    parser.add_argument("--output", "-o", help="Output CSV file (default: timestamped)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output or f"active_learning_results_{timestamp}.csv"

    main(
        benchmark_dir=args.benchmark_dir,
        num_trials=args.trials,
        output_csv=output_csv,
        parallel=not args.serial,
        num_workers=args.workers,
    )
