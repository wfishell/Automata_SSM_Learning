import json
import os
import re
import shutil
import subprocess
import tempfile

import boto3

s3 = boto3.client("s3")

ACTIVE_LEARNING_TIMEOUT = 300
SCRIPT_DIR = os.environ.get("LAMBDA_TASK_ROOT", "/var/task")


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
    subprocess.run("autfilt System.hoa --dot > System.dot", shell=True, check=True)

    ap_line = subprocess.run(
        'grep "^AP:" System.hoa', shell=True, check=True, capture_output=True, text=True
    ).stdout.strip()
    ap_names = re.findall(r'"([^"]+)"', ap_line)

    input_names_lower = {n.lower() for n in input_names}
    output_names_lower = {n.lower() for n in output_names}
    actual_inputs = [ap for ap in ap_names if ap.lower() in input_names_lower]
    actual_outputs = [ap for ap in ap_names if ap.lower() in output_names_lower]

    return ",".join(ap_names), ",".join(actual_inputs), ",".join(actual_outputs)


def Active_Learning(Original_Dot_File, Inputs, Outputs):
    script = os.path.join(SCRIPT_DIR, "active_learning.py")
    result = subprocess.run(
        f"python {script} {Original_Dot_File} --inputs {Inputs} --outputs {Outputs}",
        check=True,
        shell=True,
        capture_output=True,
        text=True,
        timeout=ACTIVE_LEARNING_TIMEOUT,
    )
    output = result.stdout + result.stderr
    num_queries = None
    match = re.search(r"Membership queries:.*=\s*(\d+)", output, re.IGNORECASE)
    if match:
        num_queries = int(match.group(1))
    subprocess.run(
        "autfilt System_learned.hoa --dot > System_learned.dot", shell=True, check=True
    )
    return num_queries, output


def Generate_And_Validate_Data(Learned_Dot_File, APs, Inputs, Outputs, Original_System):
    trace_gen = os.path.join(SCRIPT_DIR, "Dot_Trace_Generator.py")
    trace_check = os.path.join(SCRIPT_DIR, "Trace_Checker.py")
    subprocess.run(
        f"python {trace_gen} {Learned_Dot_File} --fmt dot --inputs {Inputs} --outputs {Outputs} --aps {APs} -n 1000 -l 20 --cycle --out Active_Test_Dataset.txt",
        check=True,
        shell=True,
    )
    result = subprocess.run(
        f"python {trace_check} {Original_System} Active_Test_Dataset.txt",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    accuracy = None
    match = re.search(r"Acceptance\s*%?:\s*([0-9.]+)%?", output, re.IGNORECASE)
    if match:
        accuracy = float(match.group(1))
    return accuracy, output


def lambda_handler(event, context):
    bucket = event["bucket"]
    tlsf_key = event["tlsf_key"]
    trial = event.get("trial", 1)

    work_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()

    try:
        os.chdir(work_dir)
        local_tlsf = os.path.join(work_dir, "input.tlsf")
        s3.download_file(bucket, tlsf_key, local_tlsf)

        APs, inputs, outputs = SynthesizeMealy(local_tlsf)
        num_queries, _ = Active_Learning("System.dot", inputs, outputs)
        accuracy, _ = Generate_And_Validate_Data(
            "System_learned.dot", APs, inputs, outputs, "System.hoa"
        )

        result = {
            "tlsf_file": tlsf_key,
            "trial": trial,
            "sample_size": num_queries,
            "accuracy": accuracy,
            "status": "success",
        }
    except subprocess.TimeoutExpired:
        result = {"tlsf_file": tlsf_key, "trial": trial, "status": "timeout"}
    except Exception as e:
        result = {
            "tlsf_file": tlsf_key,
            "trial": trial,
            "status": "error",
            "error": str(e),
        }
    finally:
        os.chdir(original_dir)
        shutil.rmtree(work_dir, ignore_errors=True)

    result_key = f"results/{tlsf_key.replace('/', '_')}_trial{trial}.json"
    s3.put_object(Bucket=bucket, Key=result_key, Body=json.dumps(result))
    return result
