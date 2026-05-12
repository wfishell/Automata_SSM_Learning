#!/usr/bin/env python3
"""Simulate HOA automaton on traces and report state sequences.

Usage:
    python hoa_state_tracker.py System.hoa traces.txt

Output:
    For each trace, prints the sequence of states visited.
"""

import re
import sys
from collections import defaultdict


def parse_hoa_transitions(hoa_file):
    """Parse HOA file into a transition function."""
    with open(hoa_file, "r") as f:
        content = f.read()

    n_states = int(re.search(r"States:\s*(\d+)", content).group(1))
    start_state = int(re.search(r"Start:\s*(\d+)", content).group(1))

    ap_match = re.search(r"AP:\s*\d+\s+(.*)", content)
    ap_names = re.findall(r'"([^"]+)"', ap_match.group(1))

    transitions = defaultdict(list)

    body = content.split("--BODY--")[1].split("--END--")[0]

    current_state = None
    for line in body.strip().split("\n"):
        line = line.strip()
        if line.startswith("State:"):
            current_state = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("[") and current_state is not None:
            match = re.match(r"\[([^\]]+)\]\s*(\d+)", line)
            if match:
                condition = match.group(1)
                next_state = int(match.group(2))
                transitions[current_state].append((condition, next_state))

    return n_states, start_state, ap_names, dict(transitions)


def eval_condition(condition, valuation, ap_names):
    """Evaluate a HOA condition given a valuation."""
    or_clauses = condition.split("|")

    for clause in or_clauses:
        clause = clause.strip()
        if eval_and_clause(clause, valuation, ap_names):
            return True
    return False


def eval_and_clause(clause, valuation, ap_names):
    """Evaluate a single AND clause like '!0&1&!2'."""
    terms = clause.split("&")

    for term in terms:
        term = term.strip()
        if not term:
            continue

        negated = term.startswith("!")
        if negated:
            term = term[1:]

        ap_idx = int(term)
        ap_name = ap_names[ap_idx]
        val = valuation.get(ap_name, 0)

        if negated:
            if val != 0:
                return False
        else:
            if val != 1:
                return False

    return True


def simulate_trace(trace_str, start_state, ap_names, transitions):
    """Simulate automaton on a single trace string.

    Returns:
        state_sequence: list of states visited
    """
    trace_str = trace_str.strip()
    if "cycle{" in trace_str:
        trace_str = trace_str.split("cycle{")[0].rstrip(";")

    steps = [s for s in trace_str.split(";") if s.strip()]

    state_sequence = [start_state]
    current_state = start_state

    for step in steps:
        # Parse valuation
        valuation = {}
        literals = step.split("&")
        for lit in literals:
            lit = lit.strip()
            if lit.startswith("!"):
                valuation[lit[1:]] = 0
            else:
                valuation[lit] = 1

        # Find matching transition
        found = False
        for condition, next_state in transitions.get(current_state, []):
            if eval_condition(condition, valuation, ap_names):
                current_state = next_state
                state_sequence.append(current_state)
                found = True
                break

        if not found:
            state_sequence.append(-1)  # No valid transition
            break

    return state_sequence


def main(hoa_file, trace_file):
    # Parse HOA
    n_states, start_state, ap_names, transitions = parse_hoa_transitions(hoa_file)

    print(f"HOA Automaton: {hoa_file}")
    print(f"  States: {n_states}")
    print(f"  Start: {start_state}")
    print(f"  APs: {ap_names}")
    print("=" * 60)

    # Track which states are visited
    all_states_visited = set()

    # Process traces
    with open(trace_file, "r") as f:
        traces = [line.strip() for line in f if line.strip()]

    print(f"\nProcessing {len(traces)} traces...\n")

    for i, trace in enumerate(traces):
        states = simulate_trace(trace, start_state, ap_names, transitions)
        all_states_visited.update(states)

        # Print first few traces in detail
        if i < 10:
            print(f"Trace {i}: {' -> '.join(str(s) for s in states)}")
        elif i == 10:
            print(f"... ({len(traces) - 10} more traces)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total states in automaton: {n_states}")
    print(f"States visited in traces: {sorted(all_states_visited)}")
    print(f"States NOT visited: {sorted(set(range(n_states)) - all_states_visited)}")

    # Count visits per state
    state_counts = defaultdict(int)
    for trace in traces:
        states = simulate_trace(trace, start_state, ap_names, transitions)
        for s in states:
            state_counts[s] += 1

    print("Visits per state:")
    for s in range(n_states):
        count = state_counts.get(s, 0)
        bar = "#" * min(count // 100, 50)
        print(f"  State {s:2d}: {count:6d} {bar}")

    if -1 in state_counts:
        print(f"\n  WARNING: {state_counts[-1]} invalid transitions (no matching edge)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <hoa_file> <trace_file>")
        print(f"Example: {sys.argv[0]} System.hoa Training_Dataset.txt")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
