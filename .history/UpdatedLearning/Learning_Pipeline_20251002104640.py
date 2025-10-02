import os
import sys
import argparse
import json
import random
import pandas as pd
import subprocess

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

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
