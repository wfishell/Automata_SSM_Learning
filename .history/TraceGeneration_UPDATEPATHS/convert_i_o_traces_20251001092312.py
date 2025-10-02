"""
Trace Converter + RPNI Learner
Reads traces from a text file, converts them with convert_i_o_traces_for_RPNI,
and runs RPNI to infer a Mealy/DFA.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from aalpy.learning_algs.deterministic_passive.RPNI import run_RPNI

# Import your actual conversion function


def parse_step(
    step: str, inputs: List[str], outputs: List[str]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1
    in_dict = {k: valuation.get(k, 0) for k in inputs}
    out_dict = {k: valuation.get(k, 0) for k in outputs}
    return in_dict, out_dict


def parse_trace(line: str, inputs: List[str], outputs: List[str]):
    """Parse one full trace line into [(inputs, outputs), ...]."""
    line = line.strip()
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = line.split(";")
    trace = [parse_step(s, inputs, outputs) for s in steps if s]
    return trace


def process_file(trace_file: str, inputs: List[str], outputs: List[str]):
    """Return dataset by calling convert_i_o_traces_for_RPNI on each parsed trace."""
    lines = Path(trace_file).read_text().splitlines()
    dataset = []
    for i, line in enumerate(lines):
        trace = parse_trace(line, inputs, outputs)

        # 🔹 Call your function here
        converted = convert_i_o_traces_for_RPNI(trace)

        # RPNI expects tuples: (word_as_list, label)
        # Assume all traces are positive unless you want to separate
        dataset.append((converted, True))

    return dataset


def run_rpni_learning(dataset):
    dfa = run_RPNI(dataset)
    return dfa


if __name__ == "__main__":
    # Define inputs/outputs
    INPUTS = ["cancel", "req", "go"]
    OUTPUTS = ["grant"]

    dataset = process_file("TraceGeneration_UPDATEPATHS/Result.txt", INPUTS, OUTPUTS)

    print("\n[+] Dataset (first 5 traces):")
    for d in dataset[:5]:
        print(d)

    model = run_rpni_learning(dataset)

    print("\n[+] Learned Automaton:")
    print(model)
