#!/usr/bin/env python3
"""
Moore DOT SSM - Parse Moore machine DOT files, convert to SSM, and generate traces.

Usage:
    # Just encode and print info
    python moore_ssm.py machine.dot --inputs r_0,r_1 --outputs g_0,a_0,b_0,c_0

    # Generate traces for HOA checking (Mealy-style output)
    python moore_ssm.py machine.dot --inputs r_0,r_1 --outputs g_0,a_0,b_0,c_0 \
        --test-traces -n 50 -l 15 --cycle --aps r_0,r_1,g_0,a_0,b_0,c_0 --out traces.txt

SSM dynamics (Mealy-compatible output - after transition):
    μ_t = h_t ⊗ σ_t                    # Kronecker product
    h_{t+1} = A @ h_t + B @ μ_t        # State update (optionally with tanh)
    y_t = C @ h_{t+1}                  # Output from new state
"""

import argparse
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# DOT PARSING
# =============================================================================


@dataclass
class MooreMachine:
    """Parsed Moore machine from DOT format."""

    states: List[str]
    state_to_idx: Dict[str, int]
    initial_state: str
    state_outputs: Dict[str, Dict[str, bool]]  # state -> {ap_name: value}
    transitions: Dict[str, Dict[str, str]]  # state -> {input_formula: next_state}
    input_aps: List[str]
    output_aps: List[str]


def parse_output_formula(formula: str) -> Dict[str, bool]:
    """Parse "!a_0 & !b_0 & c_0" into {ap: value}."""
    result = {}
    if not formula.strip():
        return result

    # Remove parentheses
    formula = formula.replace("(", "").replace(")", "")

    for lit in formula.split("&"):
        lit = lit.strip()
        if not lit:
            continue
        if lit.startswith("!"):
            result[lit[1:].strip()] = False
        else:
            result[lit] = True
    return result


def parse_input_formula(formula: str) -> List[Dict[str, bool]]:
    """Parse input formula into list of partial assignments (for disjunctions)."""
    formula = formula.strip()
    if formula == "1":
        return [{}]

    # Remove all parentheses - they're just for grouping, not needed for parsing
    formula = formula.replace("(", "").replace(")", "")

    results = []
    for disjunct in formula.split("|"):
        disjunct = disjunct.strip()
        if not disjunct or disjunct == "1":
            results.append({})
            continue
        assignment = {}
        for lit in disjunct.split("&"):
            lit = lit.strip()
            if not lit:
                continue
            if lit.startswith("!"):
                assignment[lit[1:].strip()] = False
            else:
                assignment[lit.strip()] = True
        results.append(assignment)
    return results


def expand_dont_cares(partial: Dict[str, bool], all_aps: List[str]) -> List[int]:
    """Expand partial assignment to all matching input indices."""
    specified = set(partial.keys())
    dont_care = [ap for ap in all_aps if ap not in specified]
    n = len(all_aps)

    indices = []
    for dc_mask in range(2 ** len(dont_care)):
        idx = 0
        dc_idx = 0
        for i, ap in enumerate(all_aps):
            bit_pos = n - 1 - i
            if ap in specified:
                if partial[ap]:
                    idx |= 1 << bit_pos
            else:
                if (dc_mask >> dc_idx) & 1:
                    idx |= 1 << bit_pos
                dc_idx += 1
        indices.append(idx)
    return indices


def parse_moore_dot(
    dot_string: str, input_aps: List[str], output_aps: List[str]
) -> MooreMachine:
    """Parse Moore machine from DOT string."""
    input_aps = sorted(input_aps)
    output_aps = sorted(output_aps)

    states = []
    state_outputs = {}
    transitions = {}
    initial_state = None

    for line in dot_string.strip().split("\n"):
        line = line.strip()

        if not line or line.startswith("digraph") or line.startswith("graph"):
            continue
        if line.startswith("rankdir") or line.startswith("node [") or line == "}":
            continue

        # Initial state: I -> m0
        if "I ->" in line:
            match = re.search(r"I\s*->\s*(\w+)", line)
            if match:
                initial_state = match.group(1)
            continue

        if re.match(r"^\s*I\s*\[", line):
            continue

        # Transition: m0 -> m1 [label="..."]
        trans_match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]*)"\]', line)
        if trans_match:
            src, dst, label = trans_match.groups()
            transitions.setdefault(src, {})[label.strip()] = dst
            continue

        # State: m0 [label="m0 | output_formula"]
        state_match = re.match(r'(\w+)\s*\[label="([^"]*)"\]', line)
        if state_match:
            state_name, label = state_match.groups()
            if "|" in label:
                _, output_formula = label.split("|", 1)
            else:
                output_formula = ""
            states.append(state_name)
            state_outputs[state_name] = parse_output_formula(output_formula)

    state_to_idx = {s: i for i, s in enumerate(states)}
    if initial_state is None and states:
        initial_state = states[0]

    return MooreMachine(
        states=states,
        state_to_idx=state_to_idx,
        initial_state=initial_state,
        state_outputs=state_outputs,
        transitions=transitions,
        input_aps=input_aps,
        output_aps=output_aps,
    )


# =============================================================================
# TRANSITION TABLE
# =============================================================================


def build_transition_table(machine: MooreMachine) -> Dict[Tuple[int, int], int]:
    """Build complete transition table: (state_idx, input_idx) -> next_state_idx."""
    table = {}
    num_inputs = 2 ** len(machine.input_aps)

    for state_name, trans_dict in machine.transitions.items():
        state_idx = machine.state_to_idx[state_name]

        for input_formula, next_state_name in trans_dict.items():
            next_idx = machine.state_to_idx[next_state_name]

            for partial in parse_input_formula(input_formula):
                for inp_idx in expand_dont_cares(partial, machine.input_aps):
                    table[(state_idx, inp_idx)] = next_idx

    # Fill in self-loops for missing transitions
    for state_idx in range(len(machine.states)):
        for inp_idx in range(num_inputs):
            if (state_idx, inp_idx) not in table:
                table[(state_idx, inp_idx)] = state_idx

    return table


# =============================================================================
# TRACE GENERATION (No torch required)
# =============================================================================


def generate_trace_from_machine(
    machine: MooreMachine,
    transition_table: Dict[Tuple[int, int], int],
    length: int,
    ap_order: List[str],
    cycle: bool = False,
) -> str:
    """Generate a trace using the Moore machine (simulating SSM behavior).

    Uses Mealy-style output: output from destination state after transition.
    This matches HOA semantics from ltlsynt.
    """
    state_idx = machine.state_to_idx[machine.initial_state]
    num_inputs = 2 ** len(machine.input_aps)

    steps = []
    for _ in range(length):
        # Random input
        input_idx = random.randint(0, num_inputs - 1)

        # Transition to next state
        next_idx = transition_table.get((state_idx, input_idx), state_idx)
        state_idx = next_idx

        # Get output from NEW state (Mealy-compatible)
        state_name = machine.states[state_idx]
        output_dict = machine.state_outputs[state_name]

        # Build input dict
        input_dict = {}
        n = len(machine.input_aps)
        for i, ap in enumerate(machine.input_aps):
            input_dict[ap] = bool((input_idx >> (n - 1 - i)) & 1)

        # Build Spot-format step
        literals = []
        for ap in ap_order:
            if ap in input_dict:
                val = input_dict[ap]
            elif ap in output_dict:
                val = output_dict[ap]
            else:
                val = False  # Default to false if not specified

            if val:
                literals.append(ap)
            else:
                literals.append(f"!{ap}")

        steps.append("&".join(literals))

    if cycle:
        steps.append("cycle{1}")

    return ";".join(steps)


# =============================================================================
# TORCH SSM MODULE (Optional)
# =============================================================================

if TORCH_AVAILABLE:

    class MooreDotSSM(nn.Module):
        """SSM encoding of Moore machine."""

        def __init__(
            self,
            machine: MooreMachine,
            transition_table: Dict,
            epsilon: float = 0.0,
            use_tanh: bool = False,
        ):
            super().__init__()
            self.machine = machine
            self.transition_table = transition_table
            self.epsilon = epsilon
            self.use_tanh = use_tanh

            self.num_states = len(machine.states)
            self.num_inputs = 2 ** len(machine.input_aps)
            self.num_output_aps = len(machine.output_aps)

            A = self._build_A()
            B = self._build_B()
            C = self._build_C()

            # Add epsilon noise to zero entries
            if epsilon > 0:
                A = self._add_epsilon_noise(A, epsilon)
                B = self._add_epsilon_noise(B, epsilon)
                C = self._add_epsilon_noise(C, epsilon)

            self.A = nn.Parameter(A)
            self.B = nn.Parameter(B)
            self.C = nn.Parameter(C)

            h0 = torch.zeros(self.num_states)
            h0[machine.state_to_idx[machine.initial_state]] = 1.0
            self.register_buffer("h0", h0)

        def _add_epsilon_noise(
            self, matrix: torch.Tensor, epsilon: float
        ) -> torch.Tensor:
            """Add uniform noise in [-epsilon, epsilon] to zero entries."""
            noise = (2 * torch.rand_like(matrix) - 1) * epsilon
            zero_mask = matrix == 0
            return matrix + noise * zero_mask.float()

        def _build_A(self) -> torch.Tensor:
            """Build A matrix (identity for state persistence)."""
            return torch.eye(self.num_states)

        def _build_B(self) -> torch.Tensor:
            N, M = self.num_states, self.num_inputs
            B = torch.zeros(N, N * M)

            for state_idx in range(N):
                for input_idx in range(M):
                    col = state_idx * M + input_idx
                    next_idx = self.transition_table.get(
                        (state_idx, input_idx), state_idx
                    )
                    if next_idx != state_idx:
                        B[state_idx, col] = -1.0
                        B[next_idx, col] = 1.0
            return B

        def _build_C(self) -> torch.Tensor:
            """Build C matrix for output mapping with one-hot output symbols.

            C ∈ ℝ^{2^num_output_aps × num_states} Each state maps to exactly one output
            symbol (one-hot).
            """
            num_output_symbols = 2**self.num_output_aps
            C = torch.zeros(num_output_symbols, self.num_states)

            for state_name, outputs in self.machine.state_outputs.items():
                state_idx = self.machine.state_to_idx[state_name]

                # Convert output AP values to symbol index
                # Index = sum(2^(n-1-i) * value[i]) for sorted output APs
                symbol_idx = 0
                n = len(self.machine.output_aps)
                for i, ap in enumerate(self.machine.output_aps):
                    if outputs.get(ap, False):
                        symbol_idx |= 1 << (n - 1 - i)

                C[symbol_idx, state_idx] = 1.0

            return C

        def output_symbol_to_aps(self, symbol_idx: int) -> Dict[str, bool]:
            """Convert one-hot output symbol index back to AP values."""
            n = len(self.machine.output_aps)
            return {
                ap: bool((symbol_idx >> (n - 1 - i)) & 1)
                for i, ap in enumerate(self.machine.output_aps)
            }

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Forward pass with Mealy-compatible output (after transition)."""
            batch_size, seq_len, _ = inputs.shape
            h = self.h0.unsqueeze(0).expand(batch_size, -1).clone()

            outputs = []
            for t in range(seq_len):
                sigma_t = inputs[:, t]
                mu_t = torch.einsum("bi,bj->bij", h, sigma_t).reshape(batch_size, -1)
                h = h @ self.A.T + mu_t @ self.B.T

                # Apply tanh activation if enabled
                if self.use_tanh:
                    h = torch.tanh(h)

                y_t = h @ self.C.T
                outputs.append(y_t)

            return torch.stack(outputs, dim=1)

        def generate_trace(
            self, length: int, ap_order: List[str], cycle: bool = False
        ) -> str:
            """Generate trace using actual SSM forward pass."""
            # Generate random input sequence
            input_indices = [
                random.randint(0, self.num_inputs - 1) for _ in range(length)
            ]

            # Build one-hot input tensor
            inputs_tensor = torch.zeros(1, length, self.num_inputs)
            for t, idx in enumerate(input_indices):
                inputs_tensor[0, t, idx] = 1.0

            # Run SSM forward pass
            with torch.no_grad():
                ssm_outputs = self(inputs_tensor)  # (1, length, 2^num_output_aps)

            # Convert to trace
            steps = []
            for t in range(length):
                # Get input values
                input_dict = {}
                n = len(self.machine.input_aps)
                for i, ap in enumerate(self.machine.input_aps):
                    input_dict[ap] = bool((input_indices[t] >> (n - 1 - i)) & 1)

                # Get output symbol from SSM (argmax of one-hot)
                output_symbol = ssm_outputs[0, t].argmax().item()
                output_dict = self.output_symbol_to_aps(output_symbol)

                # Build Spot-format step
                literals = []
                for ap in ap_order:
                    if ap in input_dict:
                        val = input_dict[ap]
                    elif ap in output_dict:
                        val = output_dict[ap]
                    else:
                        val = False

                    literals.append(ap if val else f"!{ap}")

                steps.append("&".join(literals))

            if cycle:
                steps.append("cycle{1}")

            return ";".join(steps)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Moore DOT to SSM encoder with trace generation"
    )

    parser.add_argument("dot_file", help="Moore machine DOT file")
    parser.add_argument(
        "--inputs", required=True, help="Comma-separated input APs (e.g., r_0,r_1)"
    )
    parser.add_argument(
        "--outputs",
        required=True,
        help="Comma-separated output APs (e.g., g_0,a_0,b_0,c_0)",
    )

    # Trace generation options
    parser.add_argument(
        "--test-traces",
        action="store_true",
        help="Generate traces using the encoded SSM",
    )
    parser.add_argument(
        "--aps", help="Full AP order for Spot format (required for --test-traces)"
    )
    parser.add_argument(
        "-n", "--num", type=int, default=50, help="Number of traces to generate"
    )
    parser.add_argument(
        "-l", "--length", type=int, default=15, help="Length of each trace"
    )
    parser.add_argument(
        "--cycle", action="store_true", help="Add cycle marker for infinite traces"
    )
    parser.add_argument("--out", help="Output file for traces")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Parse APs
    input_aps = [x.strip() for x in args.inputs.split(",") if x.strip()]
    output_aps = [x.strip() for x in args.outputs.split(",") if x.strip()]

    # Load and parse DOT file
    with open(args.dot_file, "r") as f:
        dot_string = f.read()

    machine = parse_moore_dot(dot_string, input_aps, output_aps)
    transition_table = build_transition_table(machine)

    print(f"Loaded Moore machine from {args.dot_file}")
    print(f"  States: {len(machine.states)}")
    print(f"  Input APs: {machine.input_aps} ({2**len(machine.input_aps)} symbols)")
    print(f"  Output APs: {machine.output_aps}")
    print(f"  Initial state: {machine.initial_state}")
    print(f"  Transitions: {sum(len(t) for t in machine.transitions.values())}")

    if args.test_traces:
        if not args.aps:
            parser.error("--aps is required when using --test-traces")

        ap_order = [x.strip() for x in args.aps.split(",") if x.strip()]

        if args.seed is not None:
            random.seed(args.seed)

        print(f"\nGenerating {args.num} traces of length {args.length}...")
        print(f"  AP order: {ap_order}")
        print(f"  Cycle: {args.cycle}")

        traces = []

        if TORCH_AVAILABLE:
            # Use actual SSM forward pass with epsilon=0 for validation
            print("  Method: *** SSM FORWARD PASS *** (PyTorch, epsilon=0)")
            ssm = MooreDotSSM(machine, transition_table, epsilon=0.0)
            print("\n  SSM matrices:")
            print(f"    A: {tuple(ssm.A.shape)}")
            print(f"    B: {tuple(ssm.B.shape)}")
            print(f"    C: {tuple(ssm.C.shape)}")
            print("\n  Running ssm.forward() for each trace...")

            for i in range(args.num):
                trace = ssm.generate_trace(args.length, ap_order, args.cycle)
                traces.append(trace)

            print(f"  Generated {args.num} traces via SSM forward pass ✓")
        else:
            # Fallback to discrete simulation
            print("  Method: Discrete simulation (PyTorch not available)")
            for _ in range(args.num):
                trace = generate_trace_from_machine(
                    machine, transition_table, args.length, ap_order, args.cycle
                )
                traces.append(trace)

        if args.out:
            with open(args.out, "w") as f:
                for t in traces:
                    f.write(t + "\n")
            print(f"\n  Written to: {args.out}")
        else:
            print("\nGenerated traces:")
            for t in traces[:5]:
                print(f"  {t[:80]}...")
            if len(traces) > 5:
                print(f"  ... ({len(traces) - 5} more)")

        return  # Exit after trace generation

    # If not generating traces, create and verify SSM
    if TORCH_AVAILABLE:
        print("\nCreating PyTorch SSM (epsilon=0 for verification)...")
        ssm = MooreDotSSM(machine, transition_table, epsilon=0.0)
        print(f"  A: {tuple(ssm.A.shape)}")
        print(f"  B: {tuple(ssm.B.shape)}")
        print(f"  C: {tuple(ssm.C.shape)}")

        # Quick verification
        print("\nVerification (5 random inputs):")
        test_seq = [random.randint(0, ssm.num_inputs - 1) for _ in range(5)]

        # Discrete simulation
        state_idx = machine.state_to_idx[machine.initial_state]
        discrete_outputs = []
        for inp_idx in test_seq:
            next_idx = transition_table.get((state_idx, inp_idx), state_idx)
            state_idx = next_idx
            state_name = machine.states[state_idx]
            out = [
                machine.state_outputs[state_name].get(ap, False)
                for ap in machine.output_aps
            ]
            discrete_outputs.append(out)

        # SSM forward
        inputs_tensor = torch.zeros(1, len(test_seq), ssm.num_inputs)
        for t, idx in enumerate(test_seq):
            inputs_tensor[0, t, idx] = 1.0

        with torch.no_grad():
            ssm_out = ssm(inputs_tensor)

        all_match = True
        for t in range(len(test_seq)):
            discrete = discrete_outputs[t]
            # Decode one-hot output symbol
            output_symbol = ssm_out[0, t].argmax().item()
            ssm_decoded = ssm.output_symbol_to_aps(output_symbol)
            ssm_binary = [ssm_decoded.get(ap, False) for ap in machine.output_aps]
            match = discrete == ssm_binary
            all_match = all_match and match
            status = "✓" if match else "✗"
            print(
                f"  t={t}: input={test_seq[t]} | discrete={discrete} | ssm={ssm_binary} {status}"
            )

        print(f"\nAll match: {all_match}")
    else:
        print("\nPyTorch not available - SSM encoding skipped")
        print("Trace generation still works without PyTorch")


if __name__ == "__main__":
    main()
