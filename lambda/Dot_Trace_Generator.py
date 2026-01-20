#!/usr/bin/env python3
import argparse
import json
import random
import re

import pydot

# ============================================================
# Loaders
# ============================================================


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_dot(path):
    graphs = pydot.graph_from_dot_file(path)
    graph = graphs[0]

    states = [
        n.get_name().strip('"')
        for n in graph.get_nodes()
        if n.get_name() not in ("node", "I")
    ]

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
        "transitions": transitions,
    }


# ============================================================
# Formula helpers
# ============================================================


def parse_formula_side(side):
    """Parse conjunctions like !a&b into {a:0,b:1}"""
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
    """Randomly choose one disjunct if '|' present."""
    expr = expr.strip("() ")
    if "|" in expr:
        choices = [c.strip() for c in expr.split("|")]
        return random.choice(choices)
    return expr


# ============================================================
# Spot conversion
# ============================================================


def step_to_spot(step, input_aps, output_aps, ap_order):
    """
    Inputs:
      - unspecified inputs: random
      - unspecified outputs: forced to 0
    """
    if "/" in step:
        inp, out = step.split("/", 1)
    else:
        inp, out = step, ""

    inp = simplify_disjunction(inp.strip())
    out = simplify_disjunction(out.strip())

    in_lits = parse_formula_side(inp)
    out_lits = parse_formula_side(out)

    valuation = {}

    # --- inputs ---
    for ap in input_aps:
        if ap in in_lits:
            valuation[ap] = in_lits[ap]
        else:
            valuation[ap] = random.randint(0, 1)

    # --- outputs ---
    for ap in output_aps:
        if ap in out_lits:
            valuation[ap] = out_lits[ap]
        else:
            valuation[ap] = 0  # FORCE outputs to 0 if unspecified

    # --- build full AP valuation ---
    bits = [ap if valuation.get(ap, 0) else f"!{ap}" for ap in ap_order]

    return "&".join(bits)


def trace_to_spot(trace, input_aps, output_aps, ap_order):
    steps = []
    cycle_part = ""

    if "cycle{" in trace:
        prefix, cycle_part = trace.split("cycle{", 1)
        cycle_part = "cycle{" + cycle_part
        raw_steps = [s for s in prefix.split(";") if s.strip()]
    else:
        raw_steps = [s for s in trace.split(";") if s.strip()]

    for st in raw_steps:
        steps.append(step_to_spot(st, input_aps, output_aps, ap_order))

    if cycle_part:
        steps.append(cycle_part)

    return ";".join(steps)


# ============================================================
# Trace generation
# ============================================================


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


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Random Mealy trace generator with Spot semantics"
    )

    parser.add_argument("file", help="Automaton (JSON or DOT)")
    parser.add_argument("--fmt", choices=["json", "dot"], required=True)

    parser.add_argument(
        "--inputs", required=True, help="Comma-separated input APs (e.g. r_0,c_0)"
    )
    parser.add_argument(
        "--outputs", required=True, help="Comma-separated output APs (e.g. g_0)"
    )

    parser.add_argument(
        "--aps", required=True, help="Full AP order for Spot (inputs+outputs)"
    )

    parser.add_argument("-n", "--num", type=int, default=5)
    parser.add_argument("-l", "--length", type=int, default=10)
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--out")

    args = parser.parse_args()

    input_aps = [x.strip() for x in args.inputs.split(",") if x.strip()]
    output_aps = [x.strip() for x in args.outputs.split(",") if x.strip()]
    ap_order = [x.strip() for x in args.aps.split(",") if x.strip()]

    machine = load_json(args.file) if args.fmt == "json" else load_dot(args.file)

    traces = []
    for _ in range(args.num):
        raw = generate_trace(machine, args.length, args.cycle)
        traces.append(trace_to_spot(raw, input_aps, output_aps, ap_order))

    if args.out:
        with open(args.out, "w") as f:
            for t in traces:
                f.write(t + "\n")
    else:
        for t in traces:
            print(t)


if __name__ == "__main__":
    main()
