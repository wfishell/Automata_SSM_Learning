import argparse
import json
import os
import random
import subprocess
import sys

import pandas as pd


def SynthesizeMealy(file_path):
    # read the file
    cmd = f'autfilt --tlsf "{file_path}" > System.hoa'
    print("Running:", cmd)

    subprocess.run('autfilt learned_mealy.hoa --dot > learned_mealy.dot',
               shell=True, check=True)
    

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
