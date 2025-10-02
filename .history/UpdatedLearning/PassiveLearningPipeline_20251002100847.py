import os
import sys
import argparse
import json
import random
import pandas as pd
import subprocess

def SynthesizeMealy(file_path):
    
    subprocess.run(f'autfilt --tlsf "{file_path}" > System.hoa',
               shell=True, check=True)
    subprocess.run(f'autfilt --tlsf "{file_path}" --dot > System.dot',
               shell=True, check=True)    

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
