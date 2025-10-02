import argparse
import json
import os
import random
import subprocess
import sys

import pandas as pd


def SynthesizeMealy(file_path):

    inputs=subprocess.run(f'syfco --print-input-signals {file_path}')
    outputs=subprocess.run(f'syfco --print-output-signals {file_path}')
    APs=inputs+outputs
    APs=APs.strip()
    inputs=inputs.strip()
    outputs=outputs.strip()
    subprocess.run(f'autfilt --tlsf "{file_path}" > System.hoa',
               shell=True, check=True)
    subprocess.run(f'autfilt --tlsf "{file_path}" --dot > System.dot',
               shell=True, check=True)    
    return APs, inputs, outputs
def GenerateTraces(Dot_File, APS, num_traces=100, trace_length=10, output_file='Training_Dataset.txt'):
    subprocess.run(f'python Dot_Trace_Generator.py System.dot -fmt dot -aps {APS} -n {num_traces} -l {trace_length} --cycle --out {output_file}',shell=True, check=True)
def CheckTraces(HOA_File, Data):
    subprocess.run(f'python Trace_Checker.py {HOA_File} {Data}')
def PassiveLearning(Data, Inputs, Outputs):
    subprocess.run(f'python Passive_Mealy_Learning.py {Data} {Inputs} {Outputs}')
    subprocess.run(f'autfilt Training_Data.hoa --dot > Training_Data.dot')
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
def PassivePipeline(TLSF_File, Num_Traces):
    APs, Inputs, Outputs = SynthesizeMealy(TLSF_File)
    GenerateTraces('System.dot',APs,)


if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
