import os
import subprocess

import pandas as pd

os.environ["MKL_THREADING_LAYER"] = "GNU"


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

    APs = ",".join([inputs, outputs]) if outputs else inputs

    subprocess.run(
        f"ltlsynt --hide-status --tlsf {file_path} > System.hoa", shell=True, check=True
    )
    subprocess.run(
        f"ltlsynt --hide-status --tlsf {file_path} --dot > System.dot",
        shell=True,
        check=True,
    )

    return APs, inputs, outputs


def GenerateTraces(
    dot_file, aps, num_traces=100, trace_length=10, output_file="Training_Dataset.txt"
):
    subprocess.run(
        [
            "python",
            "Dot_Trace_Generator.py",
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
        ["python", "Trace_Checker.py", hoa_file, data_file],
        capture_output=True,
        text=True,
        check=True,
    )
    # Look for the line with "Acceptance %:"
    acc = None
    for line in result.stdout.splitlines():
        if line.startswith("Acceptance %:"):
            acc = float(line.split(":")[1].strip().replace("%", ""))
    return acc


def pipeline(TLSF, Training_Samples, Length):
    APs, Inputs, Outputs = SynthesizeMealy(TLSF)
    GenerateTraces(
        "System.dot",
        APs,
        Training_Samples,
        trace_length=10,
        output_file="Training_Dataset.txt",
    )
    subprocess.run(
        ["python", "train_fsm_ssm.py", Inputs, Outputs],
        check=True,
    )
    Val = CheckTraces("System.hoa", "Generated_Predicted_Traces_SPOT.txt")
    return Val


if __name__ == "__main__":
    directory = "/workspaces/Automata_SSM_Learning/TestSet/SyntCompBenchMarks"
    samples = [100, 1000]
    rows = []

    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        for sample in samples:
            val = pipeline(full_path, sample, 10)
            rows.append({"file": file, "sample_size": sample, "value": val})

    df = pd.DataFrame(rows)
    df.to_csv("ssm_results.csv", index=False)
