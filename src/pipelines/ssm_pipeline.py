import os
import re
import subprocess

import pandas as pd

os.environ["MKL_THREADING_LAYER"] = "GNU"

# src/pipelines/ -> src/. Sibling scripts live in src/data/, src/scripts/.
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_GEN = os.path.join(SRC_DIR, "data", "Dot_Trace_Generator.py")
TRACE_CHECKER = os.path.join(SRC_DIR, "scripts", "Trace_Checker.py")
TRAIN_FSM_SSM = os.path.join(SRC_DIR, "scripts", "train_fsm_ssm.py")


def SynthesizeMealy(file_path):
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

    ap_line = subprocess.run(
        'grep "^AP:" System.hoa',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    ap_names = re.findall(r'"([^"]+)"', ap_line)

    input_names_lower = {n.lower() for n in input_names}
    output_names_lower = {n.lower() for n in output_names}

    actual_inputs = [ap for ap in ap_names if ap.lower() in input_names_lower]
    actual_outputs = [ap for ap in ap_names if ap.lower() in output_names_lower]

    inputs = ",".join(actual_inputs)
    outputs = ",".join(actual_outputs)
    APs = ",".join(ap_names)

    return APs, inputs, outputs


def GenerateTraces(
    dot_file, aps, num_traces=10000, trace_length=20, output_file="Training_Dataset.txt"
):
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
        if line.startswith("Acceptance %:"):
            acc = float(line.split(":")[1].strip().replace("%", ""))
    return acc


def TrainAndGetResults(inputs, outputs, training_samples):
    """Run training, monitor for convergence, return:

    - sample_complexity: training_samples * epochs_to_converge
    - final_trace_acc: test trace accuracy at convergence
    - acceptance: trace checker result on final epoch file
    - epoch_history: list of dicts with per-epoch metrics
    """
    proc = subprocess.Popen(
        ["python", "-u", TRAIN_FSM_SSM, inputs, outputs],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"      {line}", end="", flush=True)
        captured_lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)

    # Parse training output to find convergence epoch and trace accuracy
    converged_epoch = 1000
    final_trace_acc = 0.0
    epoch_history = []

    for line in captured_lines:
        # Match lines like: "Epoch 100 | Loss = 0.1234 | Train Step = 0.95 | Train Trace = 0.90 | Test Step = 0.9500 | Test Trace = 0.9000"
        match = re.search(
            r"Epoch\s+(\d+)\s*\|\s*Loss\s*=\s*([\d.]+)\s*\|.*Test Step\s*=\s*([\d.]+)\s*\|\s*Test Trace\s*=\s*([\d.]+)",
            line,
        )
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            test_step_acc = float(match.group(3))
            test_trace_acc = float(match.group(4))

            epoch_history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "test_step_acc": test_step_acc,
                    "test_trace_acc": test_trace_acc,
                }
            )

            final_trace_acc = test_trace_acc

            # Check for early convergence
            if test_trace_acc >= 1.0:
                converged_epoch = epoch
                break

    # Find the corresponding epoch file
    epoch_file = f"epoch_{converged_epoch}_test_eval.txt"

    # If exact epoch file doesn't exist, find closest one
    # (since we only write every 100 epochs)
    if not os.path.exists(epoch_file):
        # Round down to nearest 100
        closest_epoch = (converged_epoch // 100) * 100
        epoch_file = f"epoch_{closest_epoch}_test_eval.txt"

    # Run trace checker on the epoch file
    acceptance = CheckTraces("System.hoa", epoch_file)

    sample_complexity = training_samples * converged_epoch

    return {
        "converged_epoch": converged_epoch,
        "sample_complexity": sample_complexity,
        "test_trace_acc": final_trace_acc,
        "acceptance": acceptance,
        "epoch_history": epoch_history,
    }


def cleanup_epoch_files():
    """Remove epoch_*_test_eval.txt files from previous runs."""
    for f in os.listdir("."):
        if re.match(r"epoch_\d+_test_eval\.txt", f):
            os.remove(f)


def pipeline(TLSF, training_samples=10000):
    cleanup_epoch_files()

    print(f"    [1/3] Synthesizing Mealy machine from {os.path.basename(TLSF)}...", flush=True)
    APs, Inputs, Outputs = SynthesizeMealy(TLSF)
    print(f"    [2/3] Generating {training_samples} training traces...", flush=True)
    GenerateTraces(
        "System.dot",
        APs,
        training_samples,
        trace_length=20,
        output_file="Training_Dataset.txt",
    )

    print(f"    [3/3] Training SSM (streaming output below)...", flush=True)
    results = TrainAndGetResults(Inputs, Outputs, training_samples)

    print(
        f"    -> converged at epoch {results['converged_epoch']}, "
        f"test trace acc {results['test_trace_acc']:.4f}, "
        f"acceptance {results['acceptance']}",
        flush=True,
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SSM random-init training on TLSF files"
    )
    parser.add_argument(
        "--benchmark-dir",
        "-d",
        default="symbolic_vs_Gradient_learning_TLSF",
        help="Directory containing TLSF files (non-recursive)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[10000],
        help="Training sample sizes to sweep (one or more)",
    )
    parser.add_argument(
        "--summary-csv",
        default="ssm_learned_results.csv",
        help="Output CSV for per-file summary",
    )
    parser.add_argument(
        "--history-csv",
        default="ssm_epoch_history.csv",
        help="Output CSV for per-epoch history",
    )
    args = parser.parse_args()

    directory = args.benchmark_dir
    samples = args.samples

    summary_rows = []
    history_rows = []

    files = sorted(os.listdir(directory))
    print(f"Found {len(files)} files in {directory}", flush=True)
    print(f"Sweeping samples: {samples}", flush=True)
    print("=" * 60, flush=True)

    for file in files:
        full_path = os.path.join(directory, file)
        for sample in samples:
            print(f"\n>>> {file} | training_samples={sample}", flush=True)
            try:
                results = pipeline(full_path, sample)

                # Summary row
                summary_rows.append(
                    {
                        "file": file,
                        "training_samples": sample,
                        "converged_epoch": results["converged_epoch"],
                        "sample_complexity": results["sample_complexity"],
                        "test_trace_acc": results["test_trace_acc"],
                        "acceptance": results["acceptance"],
                    }
                )

                # Per-epoch history rows
                for epoch_data in results["epoch_history"]:
                    history_rows.append(
                        {
                            "file": file,
                            "training_samples": sample,
                            "epoch": epoch_data["epoch"],
                            "loss": epoch_data["loss"],
                            "test_step_acc": epoch_data["test_step_acc"],
                            "test_trace_acc": epoch_data["test_trace_acc"],
                        }
                    )

            except Exception as e:
                msg = str(e)
                if msg.startswith("[SKIP]"):
                    print(f"{file}: {msg}")
                else:
                    print(f"Error on {file} with {sample} samples: {e}")
                summary_rows.append(
                    {
                        "file": file,
                        "training_samples": sample,
                        "converged_epoch": None,
                        "sample_complexity": None,
                        "test_trace_acc": None,
                        "acceptance": None,
                    }
                )

    # Save summary results
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.summary_csv, index=False)
    print("=== Summary ===")
    print(df_summary)

    # Save per-epoch history
    df_history = pd.DataFrame(history_rows)
    df_history.to_csv(args.history_csv, index=False)
    print("\n=== Epoch History ===")
    print(df_history)
