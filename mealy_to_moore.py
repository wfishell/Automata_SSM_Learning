#!/usr/bin/env python3
"""Convert a Mealy machine (DOT format from ltlsynt) to a Moore machine.

The key insight: In a Moore machine, output depends only on state.
In a Mealy machine, output depends on state AND input.

Conversion strategy (standard algorithm):
- Moore state = (mealy_state, last_output_produced)
- The output of a Moore state is the last_output that was produced
  when transitioning INTO that state.
- Initial state needs special handling (no incoming transition).

Usage:
    python mealy_to_moore.py input.dot > output.dot
    python mealy_to_moore.py input.dot -o output.dot
"""

import argparse
import re
from collections import defaultdict


def parse_mealy_dot(dot_string):
    """Parse ltlsynt DOT output into transitions."""
    transitions = []
    initial = None

    for line in dot_string.strip().split("\n"):
        # Initial state: I -> 0
        if "I ->" in line:
            match = re.search(r"I\s*->\s*(\w+)", line)
            if match:
                initial = match.group(1)
            continue

        # Transitions: 0 -> 1 [label="!r_0 / !a_0 & !b_0 & !c_0 & !g_0"]
        match = re.search(
            r'(\w+)\s*->\s*(\w+)\s*\[label="([^/]+)\s*/\s*([^"]+)"\]', line
        )
        if match:
            src, dst, inp, out = match.groups()
            transitions.append((src.strip(), inp.strip(), out.strip(), dst.strip()))

    return initial, transitions


def mealy_to_moore(initial, transitions):
    """Convert Mealy to Moore via state splitting.

    Standard algorithm:
    - Moore state (q, o) represents: "in Mealy state q, having just produced output o"
    - The Moore output for state (q, o) is o
    - Transition: from (q1, o1) on input i, if Mealy has q1 --i/o2--> q2,
      then Moore has (q1, o1) --i--> (q2, o2)

    Initial state complication:
    - The initial Moore state hasn't produced any output yet
    - We use a distinguished "initial output" (commonly all-false or first output seen)
    """
    # Group transitions by source state
    trans_by_src = defaultdict(list)
    for src, inp, out, dst in transitions:
        trans_by_src[src].append((inp, out, dst))

    moore_states = set()  # (mealy_state, output)
    moore_transitions = []  # (src_moore, inp, dst_moore)

    # For initial state: we need Moore states for (initial, o) for each output o
    # that can be produced FROM the initial state
    initial_outputs = set()
    for inp, out, dst in trans_by_src[initial]:
        initial_outputs.add(out)

    # If no transitions from initial (edge case), use empty output
    if not initial_outputs:
        initial_outputs.add("")

    # We'll pick the lexicographically smallest as THE initial Moore state
    # but we need all of them to be reachable for complete conversion

    # Actually, the standard construction is:
    # 1. Initial Moore state is (q0, λ) where λ is a default/don't-care output
    # 2. Or: Initial Moore state is (q0, o) where o is output of first transition

    # For synthesis purposes, let's use approach where:
    # - Initial Moore state (q0, o_default) with o_default being the output
    #   that would be produced if we immediately take some transition
    # - This gives us a well-defined initial output

    # BFS to build all reachable Moore states
    visited = set()
    queue = []

    # Start with initial state - use first output as the "initial output"
    # This represents: machine starts, and its initial output is o_init
    o_init = sorted(initial_outputs)[0]  # deterministic choice
    initial_moore = (initial, o_init)
    queue.append(initial_moore)
    visited.add(initial_moore)
    moore_states.add(initial_moore)

    while queue:
        mealy_state, cur_output = queue.pop(0)

        for inp, out, dst in trans_by_src[mealy_state]:
            dst_moore = (dst, out)
            moore_transitions.append(((mealy_state, cur_output), inp, dst_moore))

            if dst_moore not in visited:
                visited.add(dst_moore)
                moore_states.add(dst_moore)
                queue.append(dst_moore)

    # Assign clean IDs to Moore states
    moore_id_map = {}  # (mealy_state, output) -> "m0", "m1", ...
    moore_outputs = {}  # "m0" -> output

    for idx, (mealy_state, output) in enumerate(sorted(moore_states)):
        mid = f"m{idx}"
        moore_id_map[(mealy_state, output)] = mid
        moore_outputs[mid] = output

    # Convert transitions to use IDs
    moore_trans_final = []
    for src_pair, inp, dst_pair in moore_transitions:
        src_id = moore_id_map[src_pair]
        dst_id = moore_id_map[dst_pair]
        moore_trans_final.append((src_id, inp, dst_id))

    initial_moore_id = moore_id_map[initial_moore]

    return moore_id_map, moore_trans_final, moore_outputs, initial_moore_id


def to_moore_dot(moore_id_map, moore_transitions, moore_outputs, initial_moore):
    """Generate DOT output for Moore machine."""
    lines = ["digraph Moore {", "  rankdir=LR", '  node [shape="circle"]']

    # Initial state marker
    if initial_moore:
        lines.append('  I [label="" style=invis width=0]')
        lines.append(f"  I -> {initial_moore}")

    # States with output labels (sorted for determinism)
    for moore_id in sorted(moore_outputs.keys(), key=lambda x: int(x[1:])):
        output = moore_outputs[moore_id]
        output_escaped = output.replace('"', '\\"')
        lines.append(f'  {moore_id} [label="{moore_id} | {output_escaped}"]')

    # Transitions (input only, deduplicated)
    seen = set()
    for src, inp, dst in sorted(moore_transitions):
        key = (src, inp, dst)
        if key not in seen:
            seen.add(key)
            inp_escaped = inp.replace('"', '\\"')
            lines.append(f'  {src} -> {dst} [label="{inp_escaped}"]')

    lines.append("}")
    return "\n".join(lines)


def convert(dot_string):
    """Main conversion function."""
    # Strip REALIZABLE/UNREALIZABLE prefix if present
    lines = dot_string.strip().split("\n")
    if lines and lines[0].strip() in ("REALIZABLE", "UNREALIZABLE"):
        dot_string = "\n".join(lines[1:])

    initial, transitions = parse_mealy_dot(dot_string)

    if not transitions:
        raise ValueError("No transitions found in DOT file")

    if initial is None:
        raise ValueError("No initial state found in DOT file")

    moore_id_map, moore_transitions, moore_outputs, initial_moore = mealy_to_moore(
        initial, transitions
    )

    return to_moore_dot(moore_id_map, moore_transitions, moore_outputs, initial_moore)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Mealy machine (DOT) to Moore machine (DOT)"
    )
    parser.add_argument("input", help="Input DOT file (Mealy machine from ltlsynt)")
    parser.add_argument("-o", "--output", help="Output DOT file (default: stdout)")

    args = parser.parse_args()

    # Read input
    with open(args.input, "r") as f:
        dot_string = f.read()

    # Convert
    moore_dot = convert(dot_string)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(moore_dot)
    else:
        print(moore_dot)


if __name__ == "__main__":
    main()
