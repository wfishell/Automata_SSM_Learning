"""
Trace Converter + RPNI Mealy Learner
Now saves the learned automaton in HOA format.
"""

from pathlib import Path
from typing import List, Dict, Tuple
from aalpy.utils import convert_i_o_traces_for_RPNI
from aalpy.learning_algs.deterministic_passive.RPNI import run_RPNI


def parse_step(step: str, inputs: List[str], outputs: List[str]) -> Tuple[Dict[str,int], str]:
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1

    in_dict = {k: valuation.get(k, 0) for k in inputs}
    out_symbol = "".join(str(valuation.get(k, 0)) for k in outputs)
    return in_dict, out_symbol


def parse_trace(line: str, inputs: List[str], outputs: List[str]):
    line = line.strip()
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = [s for s in line.split(";") if s]
    return [parse_step(s, inputs, outputs) for s in steps]

def make_prefix_closed(trace):
    """
    Convert a trace like [((1,0,0), "0"), ((1,0,1), "0"), ((0,1,1), "1")]
    into prefix-closed form suitable for RPNI.
    """
    dataset = []
    input_prefix = []
    for (inp, out) in trace:
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




def save_mealy_as_hoa(mealy, filename="learned_mealy.hoa"):
    """
    Export the learned Mealy machine to a simple HOA file.
    Each transition is labeled as "input|output".
    """
    states = list(mealy.states)
    state_ids = {s: i for i, s in enumerate(states)}

    with open(filename, "w") as f:
        f.write("HOA: v1\n")
        f.write(f"States: {len(states)}\n")
        f.write(f"Start: {state_ids[mealy.initial_state]}\n")
        f.write("AP: 0\n")  # no atomic props, we just use labels
        f.write("acc-name: all\n")
        f.write("Acceptance: 0 t\n")
        f.write("properties: explicit-labels state-acc complete\n")

        for s in states:
            sid = state_ids[s]
            for inp, (next_state, out) in s.transitions.items():
                nid = state_ids[next_state]
                label = f"{inp}|{out}"
                f.write(f"State: {sid}\n")
                f.write(f"[{label}] {nid}\n")

        f.write("--END--\n")

    print(f"[+] Saved HOA automaton to {filename}")


if __name__ == "__main__":
    INPUTS = ["cancel", "req", "go"]
    OUTPUTS = ["grant"]

    dataset = process_file("TraceGeneration_UPDATEPATHS/Result.txt", INPUTS, OUTPUTS)

    learned_mealy = run_RPNI(dataset, automaton_type="mealy")

    print("\n[+] Learned Mealy Machine:")
    print(learned_mealy)

    save_mealy_as_hoa(learned_mealy, "learned_mealy.hoa")
