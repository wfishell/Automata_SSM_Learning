import argparse
import json
import os
import random
import subprocess
import sys

import pandas as pd


def SynthesizeMealy(file_path):
    # read the file
    cmd = f'autfilt --tlsf "{f" "./PositiveTraces.txt" --dump-hoa learned_mealy.hoa'
    print("Running:", cmd)

    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
