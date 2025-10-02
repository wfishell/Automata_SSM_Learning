"""
Trace Converter + RPNI Mealy Learner
Now saves the learned automaton in HOA format.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from aalpy.learning_algs.deterministic_passive.RPNI import run_RPNI
from aalpy.utils import convert_i_o_traces_for_RPNI


def parse_step(
    step: str, inputs: List[str], outputs: List[str]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:

    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1

    # inputs as tuple of ints
    in_tuple = tuple(valuation.get(k, 0) for k in inputs)

    # outputs as tuple of ints (not strings)
    out_tuple = tuple(valuation.get(k, 0) for k in outputs)

    return in_tuple, out_tuple


def parse_trace(line: str, inputs: List[str], outputs: List[str]):
    line = line.strip()
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = [s for s in line.split(";") if s]
    return [parse_step(s, inputs, outputs) for s in steps]


def make_prefix_closed(trace):
    dataset = []
    input_prefix = []
    for inp, out in trace:
        input_prefix.append(inp)
        dataset.append((tuple(input_prefix), out))
    return dataset


def process_file(trace_file: str, inputs: List[str], outputs: List[str]):
    lines = Path(trace_file).read_text().splitlines()
    dataset = []
    for i, line in enumerate(lines):
        trace = parse_trace(line, inputs, outputs)
        prefix_closed = make_prefix_closed(trace)
        dataset.extend(prefix_closed)
        print(f"\n[Trace {i+1}] Prefix-closed expansion:")
        for ex in prefix_closed:
            print("   ", ex)
    return dataset


if __name__ == "__main__":
    INPUTS = ["cancel", "req", "go"]
    OUTPUTS = ["grant"]

    dataset = process_file("TraceGeneration_UPDATEPATHS/Result.txt", INPUTS, OUTPUTS)

    learned_mealy = run_RPNI(dataset, automaton_type="mealy")

    print("\n[+] Learned Mealy Machine:")
    print(learned_mealy)

    save_mealy_as_hoa(learned_mealy, "learned_mealy.hoa")
