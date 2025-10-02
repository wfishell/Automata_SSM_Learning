#!/usr/bin/env python3
import json
import random
import argparse
import pydot
import re

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_dot(path):
    graphs = pydot.graph_from_dot_file(path)
    graph = graphs[0]

    states = [n.get_name().strip('"') for n in graph.get_nodes() if n.get_name() not in ('node','I')]
    # find initial state via I -> X edge
    init_edges = [e for e in graph.get_edges() if e.get_source() == "I"]
    initial = init_edges[0].get_destination().strip('"') if init_edges else states[0]

    transitions = {}
    alphabet = set()

    for e in graph.get_edges():
        src = e.get_source().strip('"')
        dst = e.get_destination().strip('"')
        if src == "I":
            continue
        label = e.get_label().strip('"')
        if "/" in label:
            inp, out = label.split("/")
            inp, out = inp.strip(), out.strip()
        else:
            inp, out = label.strip(), ""
        alphabet.add(inp)
        transitions.setdefault(src, {})[inp] = (dst, out)

    return {
        "states": states,
        "initial": initial,
        "alphabet": sorted(alphabet),
        "transitions": transitions
    }

# --- Spot conversion helpers ---

def parse_formula_side(side):
    literals = {}
    if not side:
        return literals
    tokens = [tok.strip() for tok in re.split(r"&", side) if tok.strip()]
    for tok in tokens:
        if tok.startswith("!"):
            literals[tok[1:].strip()] = 0
        else:
            literals[tok.strip()] = 1
    return literals

def simplify_disjunction(expr):
    if "|" in expr:
        return expr.split("|", 1)[0].strip("() ")
    return expr.strip("() ")



def trace_to_spot(trace, ap_order):
    steps = []
    cycle_part = ""
    if "cycle{" in trace:
        prefix, cycle_part = trace.split("cycle{",1)
        cycle_part = "cycle{" + cycle_part
        raw_steps = [s for s in prefix.split(";") if s.strip()]
    else:
        raw_steps = [s for s in trace.split(";") if s.strip()]
    for st in raw_steps:
        steps.append(step_to_spot(st, ap_order))
    if cycle_part:
        steps.append(cycle_part)
    return ";".join(steps)

# --- Trace generation ---

def generate_trace(machine, length=10, cycle=False):
    state = machine["initial"]
    transitions = machine["transitions"]

    trace = []
    for _ in range(length):
        valid_inputs = list(transitions[state].keys())
        if not valid_inputs:
            break
        inp = random.choice(valid_inputs)
        next_state, out = transitions[state][inp]
        trace.append(f"{inp}/{out}")
        state = next_state

    if cycle:
        trace.append("cycle{1}")
    return ";".join(trace)

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Random trace generator with Spot semantics output.")
    parser.add_argument("file", help="Path to automaton (JSON or DOT).")
    parser.add_argument("--fmt", choices=["json","dot"], required=True, help="File format")
    parser.add_argument("--aps", required=True, help="Comma-separated list of APs in fixed order, e.g. a,b,p0,p1")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of traces")
    parser.add_argument("-l", "--length", type=int, default=10, help="Trace length")
    parser.add_argument("--cycle", action="store_true", help="Append cycle{1} at end")
    parser.add_argument("--out", help="Output file (txt). If not provided, prints to stdout.")
    args = parser.parse_args()

    ap_order = [ap.strip() for ap in args.aps.split(",") if ap.strip()]

    if args.fmt == "json":
        machine = load_json(args.file)
    else:
        machine = load_dot(args.file)

    traces = []
    for i in range(args.num):
        raw_trace = generate_trace(machine, length=args.length, cycle=args.cycle)
        traces.append(trace_to_spot(raw_trace, ap_order))

    if args.out:
        with open(args.out, "w") as f:
            for t in traces:
                f.write(t + "\n")
    else:
        for t in traces:
            print(t)

if __name__ == "__main__":
    main()