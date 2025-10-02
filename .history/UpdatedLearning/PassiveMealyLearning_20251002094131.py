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
        input_prefix = input_prefix + [inp]
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


def save_mealy_as_hoa(
    mealy, inputs: List[str], outputs: List[str], filename="learned_mealy.hoa"
):
    aps = inputs + outputs
    ap_indices = {ap: i for i, ap in enumerate(aps)}

    states = list(mealy.states)
    state_ids = {s: i for i, s in enumerate(states)}

    with open(filename, "w") as f:
        f.write("HOA: v1\n")
        f.write(f"States: {len(states)}\n")
        f.write(f"Start: {state_ids[mealy.initial_state]}\n")
        f.write(f"AP: {len(aps)} " + " ".join(f'"{ap}"' for ap in aps) + "\n")
        f.write("acc-name: all\n")
        f.write("Acceptance: 0 t\n")
        f.write("properties: trans-labels explicit-labels state-acc deterministic\n")
        f.write(
            "controllable-AP: " + " ".join(str(ap_indices[o]) for o in outputs) + "\n"
        )
        f.write("--BODY--\n")

        for s in states:
            sid = state_ids[s]
            f.write(f"State: {sid}\n")

            for inp, next_state in s.transitions.items():
                nid = state_ids[next_state]
                out = s.output_fun[inp]

                # build Boolean label over all APs
                lits = []
                for i, val in enumerate(inp + out):  # inputs first, then outputs
                    if val == 1:
                        lits.append(str(i))
                    else:
                        lits.append(f"!{i}")
                label = "&".join(lits)

                f.write(f"[{label}] {nid}\n")

        f.write("--END--\n")

    print(f"[+] Saved HOA automaton with {len(aps)} APs to {filename}")


import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_i_o_traces.py <trace_file>")
        sys.exit(1)

    trace_file = sys.argv[1]  # first argument after the script name

    INPUTS = ["r_0", "r_1"]
    OUTPUTS = ["p0", "p1"]

    dataset = process_file(trace_file, INPUTS, OUTPUTS)

    learned_mealy = run_RPNI(dataset, automaton_type="mealy")

    print("\n[+] Learned Mealy Machine:")
    print(learned_mealy)

    # use same folder as input file for output
    out_file = str(Path(trace_file).with_suffix(".hoa"))
    save_mealy_as_hoa(learned_mealy, INPUTS, OUTPUTS, out_file)
