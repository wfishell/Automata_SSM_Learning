import argparse
import json
import os
import re
import subprocess
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "GNU"


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

    subprocess.run(
        f"ltlsynt --hide-status --tlsf {file_path} > System.hoa", shell=True, check=True
    )
    subprocess.run(
        "autfilt System.hoa --dot > ./Drift_Test/Arbiter_Mealy.dot",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python mealy_to_moore.py ./Drift_Test/Arbiter_Mealy.dot -o ./Drift_Test/Arbiter_Moore.dot",
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


def GenerateTrainingData(APs, Inputs, Outputs):
    # Generate original traces
    subprocess.run(
        f"python Dot_Trace_Generator.py Drift_Test/Arbiter_Mealy.dot --fmt dot --aps {APs} -n 10000 -l 50 --cycle --out Drift_Test/Training_Dataset.txt",
        shell=True,
        check=True,
    )
    # Transform with drunk arbiter and overwrite
    subprocess.run(
        f"python Drift_Test/drunk_arbiter_data_gen.py '{Inputs}' '{Outputs}' < Drift_Test/Training_Dataset.txt > Drift_Test/Training_Dataset_tmp.txt",
        shell=True,
        check=True,
    )
    # Overwrite original with transformed
    subprocess.run(
        "mv Drift_Test/Training_Dataset_tmp.txt Drift_Test/Training_Dataset.txt",
        shell=True,
        check=True,
    )


def TrainModel(Inputs, Outputs):
    process = subprocess.Popen(
        f"python train_fsm_ssm.py {Inputs} {Outputs} ",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()
    return "".join(output_lines)


def process_single_file(tlsf_file):
    """Process a single TLSF file and return results dict."""
    results = {
        "tlsf_file": str(tlsf_file),
        "inputs": None,
        "outputs": None,
        "aps": None,
        "training_output": None,
        "error": None,
    }

    try:
        APs, inputs, outputs = SynthesizeMealy(tlsf_file)
        results["aps"] = APs
        results["inputs"] = inputs
        results["outputs"] = outputs

        GenerateTrainingData(APs, inputs, outputs)

        training_output = TrainModel(inputs, outputs)
        results["training_output"] = training_output
    except Exception as e:
        results["error"] = str(e)
        print(f"Error processing {tlsf_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train adaptive arbiter from TLSF spec(s)"
    )
    parser.add_argument(
        "path", help="Path to TLSF file or directory containing TLSF files"
    )
    parser.add_argument(
        "--output", "-o", default="training_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    path = Path(args.path)
    all_results = []

    if path.is_file():
        result = process_single_file(path)
        all_results.append(result)
    elif path.is_dir():
        tlsf_files = sorted(path.glob("*.tlsf"))
        print(f"Found {len(tlsf_files)} TLSF files in {path}")

        for i, tlsf_file in enumerate(tlsf_files):
            print(f"\n[{i+1}/{len(tlsf_files)}] Processing {tlsf_file.name}...")
            result = process_single_file(tlsf_file)
            all_results.append(result)
    else:
        print(f"Error: {path} is not a valid file or directory")
        return

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(
        f"Processed {len(all_results)} files, {sum(1 for r in all_results if r['error'] is None)} succeeded"
    )


if __name__ == "__main__":
    main()
