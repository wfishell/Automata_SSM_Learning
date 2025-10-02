"""
Trace Converter + RPNI DFA Learner
Traces are full valuations (inputs + outputs), all marked True.
Exports HOA.
"""

from pathlib import Path
from typing import List, Tuple

from aalpy.learning_algs.deterministic_passive.RPNI import run_RPNI


def parse_step(step: str, aps: List[str]) -> Tuple[int, ...]:
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1
    return tuple(valuation.get(k, 0) for k in aps)


def parse_trace(line: str, aps: List[str]):
    line = line.strip()
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = [s for s in line.split(";") if s]
    return tuple(parse_step(s, aps) for s in steps)


def process_file(trace_file: str, aps: List[str]):
    lines = Path(trace_file).read_text().splitlines()
    dataset = []
    for i, line in enumerate(lines):
        trace = parse_trace(line, aps)
        dataset.append((trace, True))  # mark everything True
        print(f"[Trace {i+1}] {trace} -> True")
    return dataset


def save_dfa_as_hoa(dfa, aps: List[str], controllable: List[str], filename="learned_dfa.hoa"):
    ap_indices = {ap: i for i, ap in enumerate(aps)}
    states = list(dfa.states)
    state_ids = {s: i for i, s in enumerate(states)}

    with open(filename, "w") as f:
        f.write("HOA: v1\n")
        f.write(f"States: {len(states)}\n")
        f.write(f"Start: {state_ids[dfa.initial_state]}\n")
        f.write(f"AP: {len(aps)} " + " ".join(f"\"{ap}\"" for ap in aps) + "\n")
        f.write("acc-name: all\n")
        f.write("Acceptance: 1 Inf(0)\n")
        f.write("properties: trans-labels explicit-labels state-acc deterministic\n")

        if controllable:
            f.write("controllable-AP: " + " ".join(str(ap_indices[o]) for o in controllable) + "\n")

        f.write("--BODY--\n")

        for s in states:
            sid = state_ids[s]
            acc = "1" if s.is_accepting else "0"
            f.write(f"State: {sid} {{{acc}}}\n")

            for sym, next_state in s.transitions.items():
                nid = state_ids[next_state]
                lits = []
                for i, val in enumerate(sym):
                    lits.append(str(i) if val == 1 else f"!{i}")
                label = "&".join(lits)
                f.write(f"[{label}] {nid}\n")

        f.write("--END--\n")

    print(f"[+] Saved HOA DFA with {len(aps)} APs to {filename}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python convert_traces_dfa.py <trace_file>")
        sys.exit(1)

    trace_file = sys.argv[1]

    INPUTS = ["a", "b"]
    OUTPUTS = ["p0", "p1"]
    APS = INPUTS + OUTPUTS  # full alphabet

    dataset = process_file(trace_file, APS)

    )

    print("\n[+] Learned DFA:")
    print(learned_dfa)

    out_file = str(Path(trace_file).with_suffix(".hoa"))
    save_dfa_as_hoa(learned_dfa, APS, OUTPUTS, out_file)
