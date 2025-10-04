#!/usr/bin/env python3
"""Convert SPOT-generated automata (from TLSF) to AALpy-compatible oracles for active
learning, learn a Mealy machine, and export both oracle and learned automata to HOA.

Usage:
    python spot_to_aalpy.py input.dot [--inputs a,b] [--outputs p0,p1] [algorithm] [eq_oracle]

Args:
    --inputs: comma-separated list of input propositions (e.g., a,b)
    --outputs: comma-separated list of output propositions (e.g., p0,p1)
    algorithm: lstar (default) or kv
    eq_oracle: random_walk (default) or w_method
"""

import argparse
import os
import re
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional

from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWalkEqOracle, WMethodEqOracle
from aalpy.SULs import SUL
from aalpy.utils import save_automaton_to_file, visualize_automaton

# -------------------------------
# SPOT DOT -> AALpy SUL wrapper
# -------------------------------


class SpotAutomatonOracle(SUL):
    """Oracle wrapper for SPOT-generated automaton.

    Implements AALpy's SUL (System Under Learning) interface.
    """

    def __init__(
        self, dot_file_path: str, input_props: List[str], output_props: List[str]
    ):
        super().__init__()
        self.dot_file = dot_file_path
        self.input_props = input_props
        self.output_props = output_props
        self.automaton = self._parse_dot_file(dot_file_path)  # keeps acceptance sets
        self.propositions = self._extract_propositions()
        self.alphabet = self._generate_input_alphabet()
        self.current_state = self.automaton["initial"]

        # Create transition lookup table for efficiency
        self.transition_map = self._build_transition_map()

        print(f"Loaded automaton from {dot_file_path}")
        print(f"  States: {len(self.automaton['states'])}")
        print(f"  Input propositions: {self.input_props}")
        print(f"  Output propositions: {self.output_props}")
        print(f"  Input alphabet size: {len(self.alphabet)}")

    def _parse_dot_file(self, filepath: str) -> Dict[str, Any]:
        """Parse SPOT-generated DOT file, keeping acceptance sets on edges."""
        with open(filepath, "r") as f:
            content = f.read()

        # States (SPOT uses numeric ids with label="<same id>")
        states = set()
        state_pattern = r'^\s*(\d+)\s+\[label="(\d+)"\]'
        for m in re.finditer(state_pattern, content, re.MULTILINE):
            states.add(m.group(1))

        # Initial state (I -> s)
        initial_pattern = r"I\s*->\s*(\d+)"
        initial_match = re.search(initial_pattern, content)
        initial_state = initial_match.group(1) if initial_match else "0"

        # Transitions: label may contain "\n{acc-sets}"
        transitions = defaultdict(list)
        trans_pattern = r'^\s*(\d+)\s*->\s*(\d+)\s*\[label="([^"]+)"[^\]]*\]'
        acc_re = re.compile(r"\{([^}]*)\}")

        for m in re.finditer(trans_pattern, content, re.MULTILINE):
            frm, to, label = m.group(1), m.group(2), m.group(3)

            cond_part, acc_part = label, ""
            if "\\n" in label:
                cond_part, acc_part = label.split("\\n", 1)
            elif "\n" in label:
                cond_part, acc_part = label.split("\n", 1)

            # Handle Mealy machine syntax: "input_condition / output_condition"
            if "/" in cond_part:
                input_cond, output_cond = cond_part.split("/", 1)
                input_cond = input_cond.strip()
                output_cond = output_cond.strip()
            else:
                input_cond = cond_part.strip()
                output_cond = None

            acc_sets = []
            acc_m = acc_re.search(acc_part or "")
            if acc_m:
                raw = acc_m.group(1).strip()
                if raw:
                    acc_sets = [
                        int(x.strip()) for x in raw.split(",") if x.strip().isdigit()
                    ]

            transitions[frm].append(
                {
                    "to": to,
                    "condition": input_cond,
                    "output_condition": output_cond,
                    "acc": acc_sets,
                }
            )

        return {
            "states": sorted(list(states)),
            "initial": initial_state,
            "transitions": dict(transitions),
        }

    def _extract_propositions(self) -> List[str]:
        """Extract atomic propositions from transition conditions."""
        props = set()
        for trans_list in self.automaton["transitions"].values():
            for trans in trans_list:
                # Extract from input condition
                cleaned = re.sub(r"[&|!() ]", " ", trans["condition"])
                found = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", cleaned)
                props.update(found)

                # Extract from output condition if present
                if trans.get("output_condition"):
                    cleaned_out = re.sub(r"[&|!() ]", " ", trans["output_condition"])
                    found_out = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", cleaned_out)
                    props.update(found_out)
        return sorted(props)

    def _generate_input_alphabet(self) -> List[str]:
        """Generate input alphabet as binary encodings of input proposition
        valuations."""
        n = len(self.input_props)
        if n > 12:
            print(f"Warning: input alphabet has {2**n} symbols (may be slow)")
        alphabet = []
        for values in product([False, True], repeat=n):
            symbol = "".join("1" if v else "0" for v in values)
            alphabet.append(symbol)
        return alphabet

    def _build_transition_map(self) -> Dict[tuple, tuple]:
        """Pre-compute transition map for all state-input pairs, returning (next_state,
        output)."""
        trans_map = {}
        for state in self.automaton["states"]:
            for symbol in self.alphabet:
                input_valuation = self._input_symbol_to_valuation(symbol)
                # Find all valid complete valuations and their destinations
                valid_transitions = self._find_valid_transitions(state, input_valuation)
                if valid_transitions:
                    # Take the first valid transition (automaton should be deterministic)
                    next_state, output_valuation = valid_transitions[0]
                    output_symbol = self._output_valuation_to_symbol(output_valuation)
                    trans_map[(state, symbol)] = (next_state, output_symbol)
                else:
                    # No valid transition, stay in current state with default output
                    default_output = "0" * len(self.output_props)
                    trans_map[(state, symbol)] = (state, default_output)
        return trans_map

    def _input_symbol_to_valuation(self, symbol: str) -> Dict[str, bool]:
        """Convert input binary string to input proposition valuation."""
        return {prop: (symbol[i] == "1") for i, prop in enumerate(self.input_props)}

    def _output_valuation_to_symbol(self, valuation: Dict[str, bool]) -> str:
        """Convert output proposition valuation to binary string."""
        return "".join(
            "1" if valuation.get(prop, False) else "0" for prop in self.output_props
        )

    def _find_valid_transitions(
        self, state: str, input_valuation: Dict[str, bool]
    ) -> List[tuple]:
        """Find all valid transitions from state with given input, checking output
        conditions."""
        valid = []

        for trans in self.automaton["transitions"].get(state, []):
            # Check if input condition matches
            if self._evaluate_condition(trans["condition"], input_valuation):
                if trans.get("output_condition"):
                    # Parse output condition to determine required output values
                    output_valuation = self._parse_output_condition(
                        trans["output_condition"]
                    )
                    if output_valuation is not None:
                        valid.append((trans["to"], output_valuation))
                else:
                    # No output condition specified, use default
                    default_output = {prop: False for prop in self.output_props}
                    valid.append((trans["to"], default_output))

        return valid

    def _parse_output_condition(self, output_cond: str) -> Optional[Dict[str, bool]]:
        """Parse output condition string to extract required output values."""
        if not output_cond or output_cond.strip() == "":
            return None

        # Try to evaluate output condition for each possible output combination
        for output_values in product([False, True], repeat=len(self.output_props)):
            output_valuation = {
                prop: val for prop, val in zip(self.output_props, output_values)
            }
            if self._evaluate_condition(output_cond, output_valuation):
                return output_valuation

        return None

    def _evaluate_condition(self, condition: str, valuation: Dict[str, bool]) -> bool:
        """Evaluate boolean condition with given valuation."""
        expr = condition

        # Replace AP names with True/False
        for prop, value in valuation.items():
            expr = re.sub(
                r"\b" + re.escape(prop) + r"\b", "True" if value else "False", expr
            )

        # Handle 1/0 constants cleanly
        expr = re.sub(r"(?<![A-Za-z0-9_])1(?![A-Za-z0-9_])", "True", expr)
        expr = re.sub(r"(?<![A-Za-z0-9_])0(?![A-Za-z0-9_])", "False", expr)

        # Convert operators
        expr = expr.replace("&", " and ").replace("|", " or ").replace("!", " not ")

        try:
            return bool(eval(expr))
        except Exception:
            print(f"Error evaluating: {condition} with {valuation}")
            return False

    # --- AALpy SUL interface ---

    def step(self, letter: str) -> str:
        if letter not in self.alphabet:
            raise ValueError(f"Invalid input: {letter}")
        next_state, output = self.transition_map.get(
            (self.current_state, letter),
            (self.current_state, "0" * len(self.output_props)),
        )
        self.current_state = next_state
        return output  # return output symbol instead of state

    def reset(self):
        self.current_state = self.automaton["initial"]

    def pre(self):
        self.reset()

    def post(self):
        pass


# -------------------------------
# Learning + Evaluation
# -------------------------------


def learn_automaton_from_spot(
    dot_file: str,
    input_props: List[str],
    output_props: List[str],
    learning_algorithm: str = "lstar",
    eq_oracle_type: str = "random_walk",
    max_rounds: int = 100,
):
    """Learn Mealy using oracle constructed from SPOT DOT."""
    oracle = SpotAutomatonOracle(dot_file, input_props, output_props)

    if len(oracle.alphabet) > 8 and eq_oracle_type == "w_method":
        print(
            f"WARNING: W-method with {len(oracle.alphabet)} symbols may be very slow!"
        )
        print("Consider using 'random_walk' instead.")

    # Equivalence oracle
    if eq_oracle_type == "random_walk":
        num_steps = min(50000, len(oracle.alphabet) * 5000)
        eq_oracle = RandomWalkEqOracle(
            alphabet=oracle.alphabet,
            sul=oracle,
            num_steps=num_steps,
            reset_prob=0.09,
            reset_after_cex=True,
        )
        print(f"Using RandomWalk with {num_steps} steps")
    elif eq_oracle_type == "w_method":
        max_states = len(oracle.automaton["states"]) * 2
        if len(oracle.alphabet) > 8:
            max_states = min(max_states, 6)
        eq_oracle = WMethodEqOracle(
            alphabet=oracle.alphabet, sul=oracle, max_number_of_states=max_states
        )
        print(f"Using W-method with max_states={max_states}")
    else:
        raise ValueError(f"Unknown equivalence oracle type: {eq_oracle_type}")

    # Learning
    print(
        f"\nStarting {learning_algorithm.upper()} learning with {eq_oracle_type} equivalence oracle..."
    )
    if learning_algorithm == "lstar":
        learned, stats = run_Lstar(
            alphabet=oracle.alphabet,
            sul=oracle,
            eq_oracle=eq_oracle,
            automaton_type="mealy",
            max_learning_rounds=max_rounds,
            print_level=2,
            cache_and_non_det_check=True,
            return_data=True,
        )
    elif learning_algorithm == "kv":
        from aalpy.learning_algs import run_KV

        learned, stats = run_KV(
            alphabet=oracle.alphabet,
            sul=oracle,
            eq_oracle=eq_oracle,
            automaton_type="mealy",
            max_learning_rounds=max_rounds,
            print_level=2,
            return_data=True,
        )

    else:
        raise ValueError(f"Unknown learning algorithm: {learning_algorithm}")

    if len(learned.states) < len(oracle.automaton["states"]):
        print(
            f"\nWARNING: Learned {len(learned.states)} < oracle {len(oracle.automaton['states'])} states."
        )
        print("Try: increase random_walk steps, or use w_method if alphabet is small.")

    return learned, oracle, stats


def compare_automata(
    learned, oracle: SpotAutomatonOracle, num_tests: int = 1000, max_length: int = 20
) -> float:
    """Quick randomized comparison of learned vs oracle outputs."""
    import random

    print("\n=== Comparing Learned vs Oracle ===")
    mismatches = 0
    example_mismatches = []

    for _ in range(num_tests):
        length = random.randint(1, max_length)
        sequence = [random.choice(oracle.alphabet) for _ in range(length)]

        # Oracle run
        oracle.reset()
        oracle_outputs = [oracle.step(sym) for sym in sequence]

        # Learned run
        learned_outputs = []
        current_state = learned.initial_state
        for sym in sequence:
            if sym in current_state.transitions:
                nxt = current_state.transitions[sym]
                # Mealy output: use output_fun
                if (
                    hasattr(current_state, "output_fun")
                    and sym in current_state.output_fun
                ):
                    out = current_state.output_fun[sym]
                else:
                    out = None
                learned_outputs.append(out)
                current_state = nxt
            else:
                learned_outputs.append(None)
                break

        if oracle_outputs != learned_outputs:
            mismatches += 1
            if len(example_mismatches) < 3:
                example_mismatches.append(
                    {
                        "sequence": sequence[:5],
                        "oracle": oracle_outputs[:5],
                        "learned": learned_outputs[:5],
                    }
                )

    if example_mismatches:
        print("Example mismatches:")
        for ex in example_mismatches:
            print(
                f"  Seq {ex['sequence']}... Oracle: {ex['oracle']}..., Learned: {ex['learned']}..."
            )

    accuracy = (num_tests - mismatches) / num_tests * 100
    print(
        f"Accuracy: {accuracy:.2f}% ({num_tests - mismatches}/{num_tests} sequences match)"
    )
    return accuracy


# -------------------------------
# HOA Exporters
# -------------------------------


def _cond_to_hoa(
    condition: str, propositions: List[str], use_indices: bool = True
) -> str:
    """Convert a SPOT-style boolean condition over named APs into HOA label syntax.

    - If use_indices=True, map AP names to numeric indices 0..n-1 (Spot prefers this).
    - Keep !, &, |, ( ), and constants t/f.
    """
    if not condition or condition.strip() == "":
        return "t"
    s = condition.strip()
    # constants
    s = re.sub(r"(?<![A-Za-z0-9_])1(?![A-Za-z0-9_])", "t", s)
    s = re.sub(r"(?<![A-Za-z0-9_])0(?![A-Za-z0-9_])", "f", s)
    if use_indices:
        for i, p in enumerate(propositions):
            s = re.sub(r"\b" + re.escape(p) + r"\b", str(i), s)
    return s


def export_spot_oracle_to_hoa(oracle: SpotAutomatonOracle, out_path: str):
    """Export the parsed SPOT automaton (transition-based acceptance) to HOA v1.

    Preserves acceptance sets found on transitions. Uses numeric AP indices.
    """
    auto = oracle.automaton
    props = oracle.propositions
    states = auto["states"]
    start = auto["initial"]

    # Identify how many acceptance sets are used
    used_sets = set()
    for trans_list in auto["transitions"].values():
        for t in trans_list:
            used_sets.update(t.get("acc", []))
    max_set = max(used_sets) if used_sets else -1
    n_sets = max_set + 1 if used_sets else 0

    lines = []
    lines.append("HOA: v1")
    lines.append(f'name: "{os.path.basename(out_path)}"')
    lines.append(f"States: {len(states)}")
    lines.append(f"Start: {start}")
    if props:
        ap_names = " ".join(f'"{p}"' for p in props)
        lines.append(f"AP: {len(props)} {ap_names}")
    else:
        lines.append("AP: 0")

    # Acceptance
    if n_sets == 0:
        lines.append("Acceptance: 0 t")
    elif n_sets == 1 and used_sets == {0}:
        lines.append("acc-name: Buchi")
        lines.append("Acceptance: 1 Inf(0)")
    else:
        lines.append(f"acc-name: generalized-Buchi {n_sets}")
        conj = " & ".join(f"Inf({i})" for i in range(n_sets))
        lines.append(f"Acceptance: {n_sets} {conj}")

    lines.append("properties: trans-labels explicit-labels")
    lines.append("--BODY--")

    for s in states:
        lines.append(f"State: {s}")
        for t in auto["transitions"].get(s, []):
            # For oracle HOA, reconstruct full condition including outputs if present
            if t.get("output_condition"):
                full_cond = f"{t['condition']} / {t['output_condition']}"
            else:
                full_cond = t["condition"]
            cond = _cond_to_hoa(full_cond, props, use_indices=True)
            to = t["to"]
            acc = t.get("acc", [])
            acc_brace = f" {{{','.join(map(str, acc))}}}" if acc else ""
            lines.append(f"[{cond}] {to}{acc_brace}")

    lines.append("--END--")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Oracle HOA saved to: {out_path}")


def export_learned_mealy_to_hoa(
    learned,
    alphabet: List[str],
    input_props: List[str],
    output_props: List[str],
    out_path: str,
    *,
    merge_by_destination: bool = True,
    label_as_indices: bool = True,
    require_deterministic: bool = True,
):
    """Export learned Mealy to HOA with both inputs and outputs as atomic propositions.

    The transition labels will include both input conditions and output values.
    """
    # Include both inputs and outputs as APs
    all_props = input_props + output_props

    # Stable mapping state -> id
    state_list = list(learned.states)
    id_map = {st: i for i, st in enumerate(state_list)}

    def make_transition_label(input_letter: str, output: str) -> str:
        """Create a label that includes both input condition and output values."""
        terms = []

        # Add input conditions
        for i, b in enumerate(input_letter):
            if i < len(input_props):
                if label_as_indices:
                    lit = str(i) if b == "1" else f"!{i}"
                else:
                    name = input_props[i]
                    lit = name if b == "1" else f"!{name}"
                terms.append(lit)

        # Add output conditions
        for i, b in enumerate(output):
            if i < len(output_props):
                prop_idx = len(input_props) + i  # offset by number of input props
                if label_as_indices:
                    lit = str(prop_idx) if b == "1" else f"!{prop_idx}"
                else:
                    name = output_props[i]
                    lit = name if b == "1" else f"!{name}"
                terms.append(lit)

        return "&".join(terms) if terms else "t"

    # HOA header
    lines = []
    lines.append("HOA: v1")
    lines.append(f'name: "{os.path.basename(out_path)}"')
    lines.append(f"States: {len(state_list)}")
    lines.append(f"Start: {id_map[learned.initial_state]}")
    if all_props:
        ap_names = " ".join(f'"{p}"' for p in all_props)
        lines.append(f"AP: {len(all_props)} {ap_names}")
    else:
        lines.append("AP: 0")
    lines.append("Acceptance: 0 t")
    lines.append("properties: trans-labels explicit-labels deterministic")
    lines.append("--BODY--")

    for st in state_list:
        sid = id_map[st]
        lines.append(f"State: {sid}")

        if merge_by_destination:
            dest_to_labels: Dict[int, List[str]] = {}
            for input_letter, nxt in st.transitions.items():
                # Get the output for this transition
                output = (
                    st.output_fun.get(input_letter, "0" * len(output_props))
                    if hasattr(st, "output_fun")
                    else "0" * len(output_props)
                )
                lbl = make_transition_label(input_letter, output)
                dest_to_labels.setdefault(id_map[nxt], []).append(lbl)

            for dest, lbls in dest_to_labels.items():
                uniq = sorted(set(lbls))
                label = uniq[0] if len(uniq) == 1 else " | ".join(uniq)
                lines.append(f"[{label}] {dest}")
        else:
            seen_pairs = set()
            for input_letter, nxt in st.transitions.items():
                output = (
                    st.output_fun.get(input_letter, "0" * len(output_props))
                    if hasattr(st, "output_fun")
                    else "0" * len(output_props)
                )
                lbl = make_transition_label(input_letter, output)
                key = (lbl, id_map[nxt])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                lines.append(f"[{lbl}] {id_map[nxt]}")

    lines.append("--END--")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Learned HOA saved to: {out_path}")


# -------------------------------
# CLI + Main
# -------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SPOT automata to AALpy and learn Mealy machines"
    )
    parser.add_argument("dot_file", help="Input DOT file from SPOT")
    parser.add_argument(
        "--inputs", help="Comma-separated input propositions (e.g., a,b)", required=True
    )
    parser.add_argument(
        "--outputs",
        help="Comma-separated output propositions (e.g., p0,p1)",
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        default="lstar",
        choices=["lstar", "kv"],
        help="Learning algorithm (default: lstar)",
    )
    parser.add_argument(
        "--eq",
        "--eq_oracle",
        default="random_walk",
        choices=["random_walk", "w_method"],
        help="Equivalence oracle (default: random_walk)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_props = [p.strip() for p in args.inputs.split(",")]
    output_props = [p.strip() for p in args.outputs.split(",")]

    print(f"Input propositions: {input_props}")
    print(f"Output propositions: {output_props}")

    # Learn automaton
    learned, oracle, stats = learn_automaton_from_spot(
        args.dot_file, input_props, output_props, args.algorithm, args.eq
    )

    queries_total = stats["queries_learning"] + stats["queries_eq_oracle"]
    print(
        f"Membership queries: {stats['queries_learning']} + {stats['queries_eq_oracle']} = {queries_total}"
    )

    print(f"Oracle states: {len(oracle.automaton['states'])}")
    print(f"Learned states: {len(learned.states)}")

    # Save learned automaton (DOT)
    learned_dot = args.dot_file.replace(".dot", "_learned.dot")
    save_automaton_to_file(learned, learned_dot)
    print(f"Learned automaton saved to: {learned_dot}")

    # Save oracle HOA next to the input DOT (preserves acceptance, numeric labels)
    oracle_hoa = args.dot_file.replace(".dot", ".hoa")
    export_spot_oracle_to_hoa(oracle, oracle_hoa)

    # Save learned HOA
    learned_hoa = args.dot_file.replace(".dot", "_learned.hoa")
    export_learned_mealy_to_hoa(
        learned,
        oracle.alphabet,
        input_props,
        output_props,
        learned_hoa,
        merge_by_destination=True,
        label_as_indices=True,
        require_deterministic=True,
    )

    # Optional: Compare accuracy on random tests for small machines
    if len(oracle.automaton["states"]) <= 10:
        compare_automata(learned, oracle)

    # Optional: Visualize (requires graphviz)
    try:
        visualize_automaton(learned, path=learned_dot.replace(".dot", ""))
        print(f"Visualization saved to: {learned_dot.replace('.dot', '.pdf')}")
    except Exception:
        print("Visualization skipped (install graphviz to enable)")


if __name__ == "__main__":
    main()
