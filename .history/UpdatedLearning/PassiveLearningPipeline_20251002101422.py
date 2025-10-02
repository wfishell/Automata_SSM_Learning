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
    subprocess.run(f'autfilt --tlsf "{file_path}" > System.hoa',
               shell=True, check=True)
    subprocess.run(f'autfilt --tlsf "{file_path}" --dot > System.dot',
               shell=True, check=True)    
    return APs
def GenerateTraces(Dot_File, APSs, num_traces=100, trace_length=10, output_file='Training_Dataset.txt'):
    subprocess.run

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
