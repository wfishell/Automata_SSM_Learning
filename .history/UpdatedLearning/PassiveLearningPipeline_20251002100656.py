import argparse
import json
import os
import random
import subprocess
import sys

import pandas as pd


def SynthesizeMealy(file_path):
    # read the file
    cmd = 
    print("Running:", cmd)

    subprocess.run(f'autfilt --tlsf "{file_path}" > System.hoa',
               shell=True, check=True)
    subprocess.run(f'autfilt --tlsf "{file_path}" > System.hoa',
               shell=True, check=True)    

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
