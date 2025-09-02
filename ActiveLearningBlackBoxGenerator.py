'''
This script is used to generate a black box for the active learning problem.
In active learning we have a hidden oracle which tells us whether a query is a positive or negative example.
Using Spot we construct DFAs from the syntcomp benchmarks and use these in our active learning process.
'''
import spot
import os
import subprocess

def ConstructDFAFromSyntcompBenchmark(file_path,output_path):
    # Ensure output directory exists
    os.makedirs("/Users/will/github/Automata_SSM_Learning/TestSet/WhiteBox/SyntCompBenchMarks", exist_ok=True)
    
    #Generate LTL formula from syntcomp benchmark
    subprocess.run(f"syfco -f ltl {file_path} > /Users/will/github/Automata_SSM_Learning/TestSet/SyntCompLTLFormulations/{output_path}.ltl", shell=True,check=True)
    #Generate DFA from LTL formula
    subprocess.run(
    f'ltl2tgba -DP --deterministic -f "$(cat /Users/will/github/Automata_SSM_Learning/TestSet/SyntCompLTLFormulations/{output_path}.ltl)" '
    f'> /Users/will/github/Automata_SSM_Learning/TestSet/WhiteBox/SyntCompBenchMarks/{output_path}.hoa',
    shell=True,
    check=True
    )
    subprocess.run(
    f'autfilt -F /Users/will/github/Automata_SSM_Learning/TestSet/WhiteBox/SyntCompBenchMarks/{output_path}.hoa -d > /Users/will/github/Automata_SSM_Learning/TestSet/WhiteBox/SyntCompBenchMarks/{output_path}.dot',
    shell=True,
    check=True
    )

if __name__ == "__main__":
    SyntCompBenchmarks = os.listdir("/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks")
    for benchmark in SyntCompBenchmarks:
        try:
            ConstructDFAFromSyntcompBenchmark(f"/Users/will/github/Automata_SSM_Learning/TestSet/SyntCompBenchMarks/{benchmark}",benchmark[:-5])
            print(f"Successfully processed {benchmark}")
        except Exception as e:
            print(f"Failed to process {benchmark}: {e}")
            continue
