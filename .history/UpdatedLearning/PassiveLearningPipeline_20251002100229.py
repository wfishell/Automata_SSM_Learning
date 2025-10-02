import os
import sys
import argparse
import json
import random
import pandas as pd

def SynthesizeMealy(file_path):
    # read the file
    with open(file_path, 'r') as file:
        content = file.read()
    # parse the file
    # synthesize the mealy machine
    # save the mealy machine

if __name__ == "__main__":
    directory='/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/'
    for file in os.listdir(directory):
