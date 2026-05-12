#!/usr/bin/env python3
"""
W-Method Conformance Testing: Test if SSM conforms to reference FSM.

Given:
- Reference FSM (from synthesized HOA file)
- Trained SSM

The W-method guarantees finding a counterexample if:
1. The reference FSM has n states
2. The SSM behaves like a machine with at most m states
3. We test all sequences in P · Σ^(m-n) · W

Usage:
    python w_method_test.py model.pt System.hoa --extra-states 2
"""

import argparse
import re
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import torch


# ============================================================
# FSM from HOA file
# ============================================================
@dataclass
class MealyFSM:
    """Mealy machine parsed from HOA file."""

    states: Set[int]
    initial: int
    ap_names: List[str]
    inputs: List[str]
    outputs: List[str]
    # transitions: (state, input_tuple) -> (next_state, output_tuple)
    transitions: Dict[Tuple[int, Tuple], Tuple[int, Tuple]]

    def step(self, state: int, input_tuple: Tuple) -> Tuple[int, Tuple]:
        """Take one step, return (next_state, output)."""
        key = (state, input_tuple)
        if key in self.transitions:
            return self.transitions[key]
        # If no exact match, try to find a matching transition
        for (s, inp), (ns, out) in self.transitions.items():
            if s == state and self._input_matches(inp, input_tuple):
                return (ns, out)
        raise ValueError(f"No transition from state {state} on input {input_tuple}")

    def _input_matches(self, pattern: Tuple, concrete: Tuple) -> bool:
        """Check if concrete input matches pattern (with don't cares as None)."""
        for p, c in zip(pattern, concrete):
            if p is not None and p != c:
                return False
        return True

    def run(self, input_sequence: List[Tuple]) -> List[Tuple]:
        """Run FSM on input sequence, return output sequence."""
        state = self.initial
        outputs = []
        for inp in input_sequence:
            state, out = self.step(state, inp)
            outputs.append(out)
        return outputs

    def get_state_after(self, input_sequence: List[Tuple]) -> int:
        """Get state after processing input sequence."""
        state = self.initial
        for inp in input_sequence:
            state, _ = self.step(state, inp)
        return state


def parse_hoa_to_fsm(hoa_path: str) -> MealyFSM:
    """Parse HOA file into MealyFSM."""
    with open(hoa_path, "r") as f:
        content = f.read()

    # Parse AP names
    ap_match = re.search(r"^AP:\s*(\d+)\s*(.*)", content, re.MULTILINE)
    ap_names = re.findall(r'"([^"]+)"', ap_match.group(2)) if ap_match else []

    # Parse controllable-AP (outputs)
    ctrl_match = re.search(r"^controllable-AP:\s*(.+)", content, re.MULTILINE)
    controllable = set()
    if ctrl_match:
        controllable = set(int(x) for x in ctrl_match.group(1).strip().split())

    outputs = [ap_names[i] for i in range(len(ap_names)) if i in controllable]
    inputs = [ap_names[i] for i in range(len(ap_names)) if i not in controllable]

    # Parse initial state
    start_match = re.search(r"^Start:\s*(\d+)", content, re.MULTILINE)
    initial = int(start_match.group(1)) if start_match else 0

    # Parse transitions
    body_match = re.search(r"--BODY--(.*)--END--", content, re.DOTALL)
    body = body_match.group(1) if body_match else ""

    states = set()
    transitions = {}

    state_blocks = re.split(r"State:\s*", body)
    for block in state_blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        state_num = int(re.match(r"(\d+)", lines[0]).group(1))
        states.add(state_num)

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            trans_match = re.match(r"\[([^\]]*)\]\s*(\d+)", line)
            if trans_match:
                label = trans_match.group(1)
                dest = int(trans_match.group(2))

                # Parse label into transitions
                parsed = parse_hoa_label(label, ap_names, inputs, outputs)
                for inp_tuple, out_tuple in parsed:
                    transitions[(state_num, inp_tuple)] = (dest, out_tuple)

    return MealyFSM(
        states=states,
        initial=initial,
        ap_names=ap_names,
        inputs=inputs,
        outputs=outputs,
        transitions=transitions,
    )


def parse_hoa_label(
    label: str, ap_names: List[str], inputs: List[str], outputs: List[str]
) -> List[Tuple[Tuple, Tuple]]:
    """Parse HOA transition label into list of (input_tuple, output_tuple).

    Handles disjunctions by returning multiple tuples.
    """
    results = []

    # Handle disjunction
    disjuncts = re.split(r"\s*\|\s*", label)

    for disj in disjuncts:
        disj = disj.strip()
        if not disj or disj == "t":
            # True - all combinations (we'll expand later)
            continue

        # Parse conjunction
        assignment = {ap: None for ap in ap_names}

        for part in re.split(r"\s*&\s*", disj):
            part = part.strip()
            if not part or part == "t":
                continue

            negated = part.startswith("!")
            if negated:
                part = part[1:]

            if part.isdigit():
                idx = int(part)
                if idx < len(ap_names):
                    assignment[ap_names[idx]] = 0 if negated else 1

        # Extract input and output
        inp_tuple = tuple(assignment.get(i) for i in inputs)
        out_tuple = tuple(assignment.get(o, 0) for o in outputs)

        # Expand don't cares in input
        expanded_inputs = expand_dont_cares(inp_tuple)
        for exp_inp in expanded_inputs:
            results.append((exp_inp, out_tuple))

    return results


def expand_dont_cares(pattern: Tuple) -> List[Tuple]:
    """Expand a pattern with None (don't care) into all concrete tuples."""
    if None not in pattern:
        return [pattern]

    results = []
    indices = [i for i, v in enumerate(pattern) if v is None]

    for combo in product([0, 1], repeat=len(indices)):
        concrete = list(pattern)
        for idx, val in zip(indices, combo):
            concrete[idx] = val
        results.append(tuple(concrete))

    return results


# ============================================================
# SSM wrapper
# ============================================================
class SSMWrapper:
    """Wrap trained SSM for conformance testing."""

    def __init__(
        self,
        model,
        input_aps: List[str],
        output_aps: List[str],
        fsm_input_order: List[str] = None,
        fsm_output_order: List[str] = None,
        device="cpu",
    ):
        self.model = model
        self.model.eval()
        self.input_aps = input_aps  # SSM's input order
        self.output_aps = output_aps  # SSM's output order
        self.device = device

        # Build reordering maps if FSM has different order
        self.input_reorder = None
        self.output_reorder = None

        if fsm_input_order and fsm_input_order != input_aps:
            # Map from FSM input index to SSM input index
            self.input_reorder = [fsm_input_order.index(ap) for ap in input_aps]
            print(f"  Input reordering: FSM {fsm_input_order} -> SSM {input_aps}")
            print(f"  Reorder map: {self.input_reorder}")

        if fsm_output_order and fsm_output_order != output_aps:
            # Map from SSM output index to FSM output index
            self.output_reorder = [output_aps.index(ap) for ap in fsm_output_order]
            print(f"  Output reordering: SSM {output_aps} -> FSM {fsm_output_order}")

        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.h = self.model.h0.unsqueeze(0).to(self.device)

    def _reorder_input(self, input_tuple: Tuple) -> Tuple:
        """Reorder input from FSM order to SSM order."""
        if self.input_reorder is None:
            return input_tuple
        # input_tuple is in FSM order, we need SSM order
        # self.input_reorder[i] = which FSM index goes to SSM index i
        result = tuple(
            input_tuple[self.input_reorder[i]] for i in range(len(self.input_aps))
        )
        return result

    def _reorder_output(self, output_tuple: Tuple) -> Tuple:
        """Reorder output from SSM order to FSM order."""
        if self.output_reorder is None:
            return output_tuple
        return tuple(
            output_tuple[self.output_reorder[i]]
            for i in range(len(self.output_reorder))
        )

    def step(self, input_tuple: Tuple) -> Tuple:
        """Take one step, return output tuple (in FSM order)."""
        with torch.no_grad():
            # Reorder input from FSM order to SSM order
            ssm_input = self._reorder_input(input_tuple)

            # Convert input to tensor
            x = torch.zeros(1, 1, len(self.input_aps), device=self.device)
            for i, val in enumerate(ssm_input):
                x[0, 0, i] = float(val if val is not None else 0)

            # Embed and transition
            x_embed = torch.relu(self.model.embed(x))
            self.h = torch.tanh(
                self.h @ self.model.A.T + x_embed[:, 0] @ self.model.B.T
            )

            # Get output
            logits = self.model.C(self.h)
            output = (logits > 0).squeeze().int().tolist()

            if isinstance(output, int):
                output = [output]

            # Reorder output from SSM order to FSM order
            return self._reorder_output(tuple(output))

    def run(self, input_sequence: List[Tuple]) -> List[Tuple]:
        """Run on input sequence, return output sequence."""
        self.reset()
        outputs = []
        for inp in input_sequence:
            out = self.step(inp)
            outputs.append(out)
        return outputs


# ============================================================
# W-Method Components
# ============================================================
def compute_state_cover(fsm: MealyFSM, alphabet: List[Tuple]) -> Dict[int, List[Tuple]]:
    """
    Compute state cover P: shortest sequences to reach each state.
    Returns dict: state -> input sequence to reach it
    """
    cover = {fsm.initial: []}
    queue = deque([(fsm.initial, [])])

    while queue and len(cover) < len(fsm.states):
        state, prefix = queue.popleft()

        for inp in alphabet:
            try:
                next_state, _ = fsm.step(state, inp)
                if next_state not in cover:
                    new_prefix = prefix + [inp]
                    cover[next_state] = new_prefix
                    queue.append((next_state, new_prefix))
            except ValueError:
                continue

    return cover


def compute_characterization_set(
    fsm: MealyFSM, alphabet: List[Tuple], max_len: int = 10
) -> List[List[Tuple]]:
    """
    Compute characterization set W: suffixes that distinguish all state pairs.
    """
    W = set()
    states = list(fsm.states)

    for i, s1 in enumerate(states):
        for s2 in states[i + 1 :]:
            # Find suffix distinguishing s1 from s2
            suffix = find_distinguishing_suffix(fsm, s1, s2, alphabet, max_len)
            if suffix is not None:
                W.add(tuple(tuple(x) for x in suffix))

    # Always include empty sequence
    W.add(())

    # Return as list of lists of tuples (tuples are hashable for dict lookup)
    return [[tuple(x) for x in w] for w in W]


def find_distinguishing_suffix(
    fsm: MealyFSM, s1: int, s2: int, alphabet: List[Tuple], max_len: int
) -> Optional[List[Tuple]]:
    """BFS to find shortest suffix that distinguishes s1 from s2."""
    queue = deque([(s1, s2, [])])
    visited = {(s1, s2)}

    while queue:
        curr1, curr2, suffix = queue.popleft()

        if len(suffix) > max_len:
            continue

        for inp in alphabet:
            try:
                next1, out1 = fsm.step(curr1, inp)
                next2, out2 = fsm.step(curr2, inp)

                if out1 != out2:
                    return suffix + [inp]

                if (next1, next2) not in visited:
                    visited.add((next1, next2))
                    queue.append((next1, next2, suffix + [inp]))
            except ValueError:
                continue

    return None


def generate_alphabet(inputs: List[str]) -> List[Tuple]:
    """Generate all possible input combinations."""
    n = len(inputs)
    return [tuple(int(b) for b in format(i, f"0{n}b")) for i in range(2**n)]


# ============================================================
# W-Method Test
# ============================================================
def w_method_test(
    fsm: MealyFSM,
    ssm: SSMWrapper,
    extra_states: int = 0,
    max_tests: int = 100,
    verbose: bool = True,
) -> Dict:
    """Run W-method conformance test.

    Args:
        fsm: Reference FSM (specification)
        ssm: SSM under test
        extra_states: m - n, how many extra states SSM might have
        max_tests: Maximum number of test sequences
        verbose: Print progress

    Returns:
        Dict with results: {conforming, counterexample, num_tests, ...}
    """
    alphabet = generate_alphabet(fsm.inputs)

    if verbose:
        print(f"Reference FSM: {len(fsm.states)} states")
        print(f"Alphabet size: {len(alphabet)}")
        print(f"Extra states assumption: {extra_states}")

    # Compute P (state cover)
    P = compute_state_cover(fsm, alphabet)
    if verbose:
        print(f"State cover P: {len(P)} prefixes")

    # Compute W (characterization set)
    W = compute_characterization_set(fsm, alphabet)
    if verbose:
        print(f"Characterization set W: {len(W)} suffixes")
        for w in W[:5]:
            print(f"  {w}")
        if len(W) > 5:
            print(f"  ... and {len(W) - 5} more")

    # Generate test sequences: P · Σ^(0..extra_states) · W
    test_count = 0
    counterexamples = []

    if verbose:
        print("\nRunning W-method tests...")

    for state, prefix in P.items():
        # Middle part: Σ^0, Σ^1, ..., Σ^extra_states
        for mid_len in range(extra_states + 1):
            for middle in product(alphabet, repeat=mid_len):
                for w in W:
                    # Ensure all elements are tuples
                    test_seq = (
                        [tuple(x) for x in prefix]
                        + [tuple(x) for x in middle]
                        + [tuple(x) for x in w]
                    )

                    if test_count >= max_tests:
                        if verbose:
                            print(f"Reached max tests ({max_tests})")
                        return {
                            "conforming": len(counterexamples) == 0,
                            "counterexamples": counterexamples,
                            "num_tests": test_count,
                            "max_reached": True,
                        }

                    # Run both machines
                    fsm_output = fsm.run(test_seq)
                    ssm_output = ssm.run(test_seq)

                    test_count += 1

                    if fsm_output != ssm_output:
                        ce = {
                            "sequence": test_seq,
                            "fsm_output": fsm_output,
                            "ssm_output": ssm_output,
                            "prefix": prefix,
                            "middle": list(middle),
                            "suffix": w,
                            "first_diff": next(
                                (
                                    i
                                    for i, (f, s) in enumerate(
                                        zip(fsm_output, ssm_output)
                                    )
                                    if f != s
                                ),
                                len(fsm_output),
                            ),
                        }
                        counterexamples.append(ce)

                        if verbose:
                            print(f"\n{'='*60}")
                            print(f"COUNTEREXAMPLE FOUND at test {test_count}!")
                            print(f"{'='*60}")
                            print(f"Input sequence ({len(test_seq)} steps):")
                            for i, inp in enumerate(test_seq[:10]):
                                # Show in FSM order (how test was generated)

                                # Show in SSM order (how SSM received it)
                                ssm_inp = ssm._reorder_input(inp)
                                ssm_inp_str = " & ".join(
                                    ap if v else f"!{ap}"
                                    for ap, v in zip(ssm.input_aps, ssm_inp)
                                )
                                print(f"  [{i}] {ssm_inp_str}")
                            if len(test_seq) > 10:
                                print(f"  ... ({len(test_seq) - 10} more)")

                            print(f"\nFirst difference at step {ce['first_diff']}:")
                            print(f"  FSM output: {fsm_output[ce['first_diff']]}")
                            print(f"  SSM output: {ssm_output[ce['first_diff']]}")

                        # Continue to find more counterexamples or stop here
                        if len(counterexamples) >= 10:
                            return {
                                "conforming": False,
                                "counterexamples": counterexamples,
                                "num_tests": test_count,
                                "max_reached": False,
                            }

    if verbose:
        print(f"\nCompleted {test_count} tests")
        if counterexamples:
            print(f"Found {len(counterexamples)} counterexamples")
        else:
            print("NO COUNTEREXAMPLES FOUND - SSM conforms to FSM!")

    return {
        "conforming": len(counterexamples) == 0,
        "counterexamples": counterexamples,
        "num_tests": test_count,
        "max_reached": False,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="W-method conformance testing for SSM")
    parser.add_argument("model_path", type=str, help="Path to saved SSM model (.pt)")
    parser.add_argument("hoa_path", type=str, help="Path to reference FSM (.hoa)")
    parser.add_argument(
        "--extra-states",
        type=int,
        default=0,
        help="Extra states assumption (m - n, default: 0)",
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        default=100000,
        help="Maximum test sequences (default: 100000)",
    )
    args = parser.parse_args()

    # Load reference FSM
    print(f"Loading reference FSM from {args.hoa_path}...")
    fsm = parse_hoa_to_fsm(args.hoa_path)
    print(f"  States: {len(fsm.states)}")
    print(f"  Inputs: {fsm.inputs}")
    print(f"  Outputs: {fsm.outputs}")

    # Load SSM
    print(f"\nLoading SSM from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location="cpu")

    from State_Space_Model import FSM_SSM

    model = FSM_SSM(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        state_dim=checkpoint["state_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    ssm = SSMWrapper(
        model,
        input_aps=checkpoint["input_aps"],
        output_aps=checkpoint["output_aps"],
        fsm_input_order=fsm.inputs,
        fsm_output_order=fsm.outputs,
    )
    print(f"  State dim: {checkpoint['state_dim']}")

    # Verify AP alignment
    if set(fsm.inputs) != set(checkpoint["input_aps"]):
        print(f"  FSM inputs: {fsm.inputs}")
        print(f"  SSM inputs: {checkpoint['input_aps']}")

    if set(fsm.outputs) != set(checkpoint["output_aps"]):
        print("\nWARNING: Output AP mismatch!")
        print(f"  FSM outputs: {fsm.outputs}")
        print(f"  SSM outputs: {checkpoint['output_aps']}")

    # Run W-method test
    print(f"\n{'='*60}")
    print("W-METHOD CONFORMANCE TEST")
    print(f"{'='*60}")

    results = w_method_test(
        fsm, ssm, extra_states=args.extra_states, max_tests=args.max_tests, verbose=True
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {results['num_tests']}")
    print(f"Counterexamples: {len(results['counterexamples'])}")
    print(f"Conforming: {'YES' if results['conforming'] else 'NO'}")

    if results["conforming"]:
        n = len(fsm.states)
        m = n + args.extra_states
        print(f"\nThe SSM conforms to the {n}-state FSM")
        print(f"(assuming SSM has at most {m} effective states)")
    else:
        print("\nSSM does NOT conform to the FSM specification!")
        print("First counterexample:")
        ce = results["counterexamples"][0]
        print(f"  Sequence length: {len(ce['sequence'])}")
        print(f"  First difference at step: {ce['first_diff']}")


if __name__ == "__main__":
    main()
