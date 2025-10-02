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


def load_dot(path):
    (graph,) = pydot.graph_from_dot_file(path)
    G = nx.drawing.nx_pydot.from_pydot(graph)

    states = list(G.nodes())
    # infer initial: invisible node "I" pointing to one state
    init_edges = [(u, v) for u, v in G.edges() if u == "I"]
    initial = init_edges[0][1] if init_edges else states[0]

    alphabet = set()
    transitions = {}
    for u, v, d in G.edges(data=True):
        if u == "I":
            continue
        label = d.get("label", "").strip('"')
        if "/" in label:
            inp, out = label.split("/")
            inp, out = inp.strip(), out.strip()
        else:
            inp, out = label, ""
        alphabet.add(inp)
        transitions.setdefault(u, {})[inp] = (v, out)

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
