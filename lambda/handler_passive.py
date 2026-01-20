import json
import os
import re
import shutil
import subprocess
import tempfile

import boto3

s3 = boto3.client("s3")

SCRIPT_DIR = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")
TRACE_COUNTS = [5000, 10000, 15000, 20000, 25000, 30000]


def SynthesizeMealy(file_path, work_dir):
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

    hoa_file = os.path.join(work_dir, "System.hoa")
    dot_file = os.path.join(work_dir, "System.dot")

    subprocess.run(
        f"ltlsynt --hide-status --tlsf {file_path} > {hoa_file}", shell=True, check=True
    )
    subprocess.run(f"autfilt {hoa_file} --dot > {dot_file}", shell=True, check=True)

    ap_line = subprocess.run(
        f'grep "^AP:" {hoa_file}',
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

    return (
        ",".join(ap_names),
        ",".join(actual_inputs),
        ",".join(actual_outputs),
        hoa_file,
        dot_file,
    )


def GenerateTraces(
    dot_file, aps, inputs, outputs, num_traces, trace_length, output_file
):
    """Generate traces from a DOT file with explicit input/output semantics."""
    script_path = os.path.join(SCRIPT_DIR, "Dot_Trace_Generator.py")
    subprocess.run(
        [
            "python",
            script_path,
            dot_file,
            "--fmt",
            "dot",
            "--inputs",
            inputs,
            "--outputs",
            outputs,
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
    script_path = os.path.join(SCRIPT_DIR, "Trace_Checker.py")
    result = subprocess.run(
        ["python", script_path, hoa_file, data_file],
        capture_output=True,
        text=True,
        check=True,
    )
    acc = None
    for line in result.stdout.splitlines():
        if "Acceptance" in line and "%" in line:
            match = re.search(r"([0-9.]+)%?", line.split(":")[-1])
            if match:
                acc = float(match.group(1))
    return acc


def PassiveLearning(data_file, inputs, outputs, work_dir):
    script_path = os.path.join(SCRIPT_DIR, "Passive_Mealy_Learning.py")
    learned_hoa = os.path.join(work_dir, "Training_Dataset.hoa")
    learned_dot = os.path.join(work_dir, "Training_Dataset.dot")

    subprocess.run(
        ["python", script_path, data_file, inputs, outputs], check=True, cwd=work_dir
    )
    subprocess.run(
        f"autfilt {learned_hoa} --dot > {learned_dot}", shell=True, check=True
    )
    return learned_dot


def lambda_handler(event, context):
    bucket = event["bucket"]
    tlsf_key = event["tlsf_key"]
    trial = event.get("trial", 1)
    trace_length = event.get("trace_length", 20)
    test_traces = event.get("test_traces", 1000)

    work_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()

    results_by_trace_count = []

    try:
        os.chdir(work_dir)
        local_tlsf = os.path.join(work_dir, "input.tlsf")
        s3.download_file(bucket, tlsf_key, local_tlsf)

        APs, inputs, outputs, hoa_file, dot_file = SynthesizeMealy(local_tlsf, work_dir)

        for num_traces in TRACE_COUNTS:
            train_file = os.path.join(work_dir, f"Training_{num_traces}.txt")
            test_file = os.path.join(work_dir, f"Test_{num_traces}.txt")

            # Generate training traces from ground truth
            GenerateTraces(
                dot_file, APs, inputs, outputs, num_traces, trace_length, train_file
            )

            # Learn automaton from training traces
            learned_dot = PassiveLearning(train_file, inputs, outputs, work_dir)

            # Generate test traces from learned automaton
            GenerateTraces(
                learned_dot, APs, inputs, outputs, test_traces, trace_length, test_file
            )

            # Check test traces against ground truth
            accuracy = CheckTraces(hoa_file, test_file)

            results_by_trace_count.append(
                {
                    "num_traces": num_traces,
                    "sample_size": num_traces * trace_length,
                    "accuracy": accuracy,
                }
            )

            # Stop early if 100% accuracy
            if accuracy is not None and accuracy >= 100.0:
                break

        result = {
            "tlsf_file": tlsf_key,
            "trial": trial,
            "trace_length": trace_length,
            "results": results_by_trace_count,
            "final_accuracy": (
                results_by_trace_count[-1]["accuracy"]
                if results_by_trace_count
                else None
            ),
            "final_traces": (
                results_by_trace_count[-1]["num_traces"]
                if results_by_trace_count
                else None
            ),
            "status": "success",
        }

    except subprocess.TimeoutExpired:
        result = {
            "tlsf_file": tlsf_key,
            "trial": trial,
            "results": results_by_trace_count,
            "status": "timeout",
        }
    except Exception as e:
        result = {
            "tlsf_file": tlsf_key,
            "trial": trial,
            "results": results_by_trace_count,
            "status": "error",
            "error": str(e),
        }
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

    result_key = f"results-passive/{tlsf_key.replace('/', '_')}_trial{trial}.json"
    s3.put_object(Bucket=bucket, Key=result_key, Body=json.dumps(result))
    return result
