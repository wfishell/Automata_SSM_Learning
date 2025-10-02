#!/usr/bin/env python3
import argparse
import json
import random

import networkx as nx
import pydot


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


import pydot


def load_dot(path):
    graphs = pydot.graph_from_dot_file(path)
    graph = graphs[0]

    states = [
        n.get_name().strip('"')
        for n in graph.get_nodes()
        if n.get_name() not in ("node", "I")
    ]
    # find initial state via I -> X edge
    init_edges = [e for e in graph.get_edges() if e.get_source() == "I"]
    initial = init_edges[0].get_destination().strip('"') if init_edges else states[0]

    transitions = {}
    alphabet = set()

    for e in graph.get_edges():
        src = e.get_source().strip('"')
        dst = e.get_destination().strip('"')
        if src == "I":  # skip dummy init edge
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
        "transitions": transitions,
    }


def generate_trace(machine, length=10, cycle=False):
    state = machine["initial"]
    alphabet = machine["alphabet"]
    transitions = machine["transitions"]

    trace = []
    for _ in range(length):
        inp = random.choice(alphabet)
        if inp not in transitions[state]:
            # dead end for this input, stop
            break
        next_state, out = transitions[state][inp]
        trace.append(f"{inp}/{out}")
        state = next_state

    if cycle:
        trace.append("cycle{1}")
    return ";".join(trace)


def main():
    parser = argparse.ArgumentParser(
        description="Random trace generator for deterministic Mealy machines."
    )
    parser.add_argument("file", help="Path to automaton (JSON or DOT).")
    parser.add_argument(
        "--fmt", choices=["json", "dot"], required=True, help="File format"
    )
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of traces")
    parser.add_argument("-l", "--length", type=int, default=10, help="Trace length")
    parser.add_argument("--cycle", action="store_true", help="Append cycle{1} at end")
    args = parser.parse_args()

    if args.fmt == "json":
        machine = load_json(args.file)
    else:
        machine = load_dot(args.file)

    for i in range(args.num):
        print(generate_trace(machine, length=args.length, cycle=args.cycle))


if __name__ == "__main__":
    main()
