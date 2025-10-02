#!/usr/bin/env python3
import os
import subprocess
import sys
import shlex
from pathlib import Path


def get_tlsf_info(pathname):  # include file extension
    ins = subprocess.run(
        ["syfco", "--print-input-signals", pathname],
        capture_output=True, text=True, check=True
    ).stdout.replace(" ", "").strip()

    outs = subprocess.run(
        ["syfco", "--print-output-signals", pathname],
        capture_output=True, text=True, check=True
    ).stdout.replace(" ", "").strip()

    ltl = subprocess.run(
        ["syfco", "-f", "ltl", "-m", "fully", pathname],
        capture_output=True, text=True, check=True
    ).stdout.strip()

    return ins, outs, ltl

def DotAndHoa(tlsf_pathname):
    inputs, outputs, ltl = get_tlsf_info(tlsf_pathname)
    cmd = [
        "python", "-u", "GenerateDotAndHoa.py",
        "--inputs", inputs,
        "--outputs", outputs,
        "--formula", ltl,
        "--hoa", "controller.hoa",  # Ensure GenerateDotAndHoa.py uses ltlsynt --hoaf internally
    ]
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
def HOAX_Generator(controller='controller.hoa',number_of_samples=10):
    cmd = f"python {controller} Random_Trace_Generator.toml {number_of_samples}"
    subprocess.run(cmd,check=True, stdout=sys.stdout,stderr=sys.stderr)
def Check_Traces(tlsf_pathname):
    cmd = f"python check_traces_spot.py {tlsf_pathname} test.hoa  HOA_Traces SpotTraces"
    subprocess.run(cmd,check=True, stdout=sys.stdout,stderr=sys.stderr)
def PassiveLearningDataGeneration():
    cmd=f'python3 PassiveLearningTraceGenerator.py SpotTraces PositiveTraceTrainingData.txt'
    subprocess.run(cmd,check=True, stdout=sys.stdout,stderr=sys.stderr)
def PassiveLearning(tlsf_file,DataSet='./PositiveTraceTrainingData.txt'):
    inputs = subprocess.run(
        ["syfco", "--print-input-signals", tlsf_file],
        capture_output=True, text=True, check=True
    ).stdout.replace(" ", "").strip()

    outputs = subprocess.run(
        ["syfco", "--print-output-signals", tlsf_file],
        capture_output=True, text=True, check=True
    ).stdout.replace(" ", "").strip()


    cmd = f'python Passive_Learning.py "{inputs}" "{outputs}" "./PositiveTraces.txt" --dump-hoa learned_mealy.hoa'
    print("Running:", cmd)

    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    if result.returncode != 0:
        print("Error during passive learning:")
        print(result.stderr)
    else:
        print("Passive learning completed successfully.")
        print(result.stdout)
        print(f"HOA file saved to: learned_mealy.hoa")
    subprocess.run('autfilt learned_mealy.hoa --dot > learned_mealy.dot',
               shell=True, check=True)
def ActiveLearning(tlsf_file):
    inputs = subprocess.run(
        ["syfco", "--print-input-signals", tlsf_file],
        capture_output=True, text=True, check=True
    ).stdout.replace(" ", "").strip()

    outputs = subprocess.run(
        ["syfco", "--print-output-signals", tlsf_file],
        capture_output=True, text=True, check=True
    ).stdout.replace(" ", "").strip()
    
    result = subprocess.run(
        ['python', 'ActiveLearning.py', 'controller.dot', 
         "--inputs", inputs, "--outputs", outputs, 
         '--algorithm', 'lstar', '--eq', 'random_walk'],
        capture_output=True, text=True, check=True
    )
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    subprocess.run('autfilt controller_learned.hoa --dot > controller_learned.dot',
               shell=True, check=True)

if __name__=='__main__':
    tlsf = '/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/Test11.tlsf'
    # 1) Build controller.* (ensure GenerateDotAndHoa.py is NOT duplicated)
    DotAndHoa(tlsf)
    print('Construct master hoa and dot files representing true system behavior')
    print('Saved to controller.dot and controller.hoa')
    # 2) Generate traces
    HOAX_Generator()
    Check_Traces(tlsf)
    PassiveLearningDataGeneration()
    print('Training Data constructed and saved to PositiveTraces.txt')
    # 3) Check traces
    print('if not 0% something wrong with data generation or hoa')
    PassiveLearning(tlsf)
    print('Trained Model saved to learned_mealy.hoa')

    print('-----------------------------------------------------------')
    ActiveLearning(tlsf)
    print('actively learning')
    #GenerateTestTraces(tlsf, Controller='controller_learned.dot',nums=20,output_file='ActiveTestTraces.txt')
    #CheckTraces(tlsf,DataSet='./ActiveTestTraces.txt')





