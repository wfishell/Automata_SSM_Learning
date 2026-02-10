import re
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

# =============================================================================
# HOA PARSING
# =============================================================================


@dataclass
class ParsedTransition:
    from_state: int
    to_state: int
    input_values: dict  # ap_index -> bool (for input APs)
    output_values: dict  # ap_index -> bool (for output APs)


@dataclass
class ParsedHOA:
    num_states: int
    start_state: int
    ap_names: list  # List of AP names in order
    input_aps: list  # Indices of input APs
    output_aps: list  # Indices of output (controllable) APs
    transitions: list  # List of ParsedTransition


def parse_hoa(hoa_string: str) -> ParsedHOA:
    """Parse HOA format automaton specification."""
    lines = hoa_string.strip().split("\n")

    num_states = None
    start_state = None
    ap_names = []
    output_aps = []
    transitions = []
    in_body = False
    current_state = None

    for line in lines:
        line = line.strip()

        if line.startswith("States:"):
            num_states = int(line.split(":")[1].strip())
        elif line.startswith("Start:"):
            start_state = int(line.split(":")[1].strip())
        elif line.startswith("AP:"):
            ap_names = re.findall(r'"([^"]+)"', line)
        elif line.startswith("controllable-AP:"):
            parts = line.split(":")[1].strip().split()
            output_aps = [int(x) for x in parts]
        elif line == "--BODY--":
            in_body = True
        elif line == "--END--":
            in_body = False
        elif in_body and line.startswith("State:"):
            current_state = int(line.split(":")[1].strip())
        elif in_body and line.startswith("["):
            match = re.match(r"\[([^\]]+)\]\s+(\d+)", line)
            if match:
                guard_str = match.group(1)
                to_state = int(match.group(2))

                # Parse each disjunct (separated by |)
                for disjunct in guard_str.split("|"):
                    disjunct = disjunct.strip()
                    input_values = {}
                    output_values = {}

                    # Parse each conjunct (separated by &)
                    for conj in disjunct.split("&"):
                        conj = conj.strip()
                        negated = conj.startswith("!")
                        ap_idx = int(conj.lstrip("!"))
                        value = not negated

                        if ap_idx in output_aps:
                            output_values[ap_idx] = value
                        else:
                            input_values[ap_idx] = value

                    transitions.append(
                        ParsedTransition(
                            from_state=current_state,
                            to_state=to_state,
                            input_values=input_values,
                            output_values=output_values,
                        )
                    )

    input_aps = [i for i in range(len(ap_names)) if i not in output_aps]

    return ParsedHOA(
        num_states=num_states,
        start_state=start_state,
        ap_names=ap_names,
        input_aps=input_aps,
        output_aps=output_aps,
        transitions=transitions,
    )


# =============================================================================
# MEALY TO MOORE CONVERSION
# =============================================================================


@dataclass
class MooreMachine:
    """Moore machine converted from Mealy.

    Each Moore state = (mealy_state, output_tuple) The output_tuple is what was produced
    on the transition INTO this state.
    """

    num_states: int
    num_inputs: int  # 2^|input_aps|
    num_output_aps: int
    start_state: int
    transitions: dict  # (moore_state_idx, input_idx) -> next_moore_state_idx
    state_outputs: list  # moore_state_idx -> list of output AP values
    input_ap_indices: list  # Sorted list of input AP indices
    output_ap_indices: list  # Sorted list of output AP indices
    moore_to_mealy: dict  # moore_idx -> (mealy_state, output_tuple)
    mealy_to_moore: dict  # (mealy_state, output_tuple) -> moore_idx
    # Store the Mealy transition table for verification
    mealy_transitions: dict  # (mealy_state, input_idx) -> (next_mealy, output_tuple)


def mealy_to_moore(hoa: ParsedHOA) -> MooreMachine:
    """Convert Mealy machine to Moore machine by state expansion.

    Moore state = (Mealy state, output produced when ENTERING this state)
    """
    num_inputs = 2 ** len(hoa.input_aps)
    num_output_aps = len(hoa.output_aps)
    sorted_output_aps = sorted(hoa.output_aps)
    sorted_input_aps = sorted(hoa.input_aps)

    # Step 1: Build complete Mealy transition table
    # (mealy_state, input_idx) -> (next_mealy_state, output_tuple)
    mealy_transitions = {}

    for trans in hoa.transitions:
        # Find which input APs are specified vs don't-care
        specified_inputs = set(trans.input_values.keys())
        all_input_aps = set(hoa.input_aps)
        dont_care_aps = sorted(all_input_aps - specified_inputs)

        # Get output tuple (ordered by sorted output AP indices)
        output_tuple = tuple(
            trans.output_values.get(ap, False) for ap in sorted_output_aps
        )

        # Expand don't-cares
        for dc_mask in range(2 ** len(dont_care_aps)):
            full_input_idx = 0
            dc_idx = 0

            for i, ap_idx in enumerate(sorted_input_aps):
                bit_pos = len(sorted_input_aps) - 1 - i

                if ap_idx in specified_inputs:
                    if trans.input_values[ap_idx]:
                        full_input_idx |= 1 << bit_pos
                else:
                    if (dc_mask >> dc_idx) & 1:
                        full_input_idx |= 1 << bit_pos
                    dc_idx += 1

            key = (trans.from_state, full_input_idx)
            if key not in mealy_transitions:
                mealy_transitions[key] = (trans.to_state, output_tuple)

    # Step 2: BFS to find all reachable Moore states
    # Start state has initial output (all False) - this is a dummy output
    # since no transition has been taken yet
    initial_output = tuple([False] * num_output_aps)

    moore_states = set()
    moore_states.add((hoa.start_state, initial_output))
    queue = [(hoa.start_state, initial_output)]

    while queue:
        mealy_state, _ = queue.pop(0)

        for input_idx in range(num_inputs):
            key = (mealy_state, input_idx)
            if key in mealy_transitions:
                next_mealy, output = mealy_transitions[key]
                moore_state = (next_mealy, output)

                if moore_state not in moore_states:
                    moore_states.add(moore_state)
                    queue.append(moore_state)

    # Step 3: Assign deterministic indices to Moore states
    moore_states = sorted(moore_states)
    moore_to_idx = {ms: i for i, ms in enumerate(moore_states)}
    idx_to_moore = {i: ms for ms, i in moore_to_idx.items()}

    num_moore_states = len(moore_states)
    start_moore = (hoa.start_state, initial_output)
    start_idx = moore_to_idx[start_moore]

    # Step 4: Build Moore transitions
    moore_transitions = {}

    for moore_state in moore_states:
        mealy_state, _ = moore_state
        moore_idx = moore_to_idx[moore_state]

        for input_idx in range(num_inputs):
            key = (mealy_state, input_idx)
            if key in mealy_transitions:
                next_mealy, output = mealy_transitions[key]
                next_moore = (next_mealy, output)
                next_moore_idx = moore_to_idx[next_moore]
                moore_transitions[(moore_idx, input_idx)] = next_moore_idx

    # Step 5: Build state outputs
    state_outputs = []
    for i in range(num_moore_states):
        _, output_tuple = idx_to_moore[i]
        state_outputs.append(list(output_tuple))

    print("\nMealy-to-Moore conversion:")
    print(f"  Mealy states: {hoa.num_states}")
    print(f"  Moore states: {num_moore_states}")
    print(f"  Start state: {start_idx} = {idx_to_moore[start_idx]}")

    return MooreMachine(
        num_states=num_moore_states,
        num_inputs=num_inputs,
        num_output_aps=num_output_aps,
        start_state=start_idx,
        transitions=moore_transitions,
        state_outputs=state_outputs,
        input_ap_indices=sorted_input_aps,
        output_ap_indices=sorted_output_aps,
        moore_to_mealy=idx_to_moore,
        mealy_to_moore=moore_to_idx,
        mealy_transitions=mealy_transitions,
    )


# =============================================================================
# HOA SSM MODULE
# =============================================================================


class HOA_SSM(nn.Module):
    """State Space Model encoding of a Mealy machine (via Moore conversion).

    The key is output timing:
    - Trace data: at time t, we see (input_t, output_t) where output_t is
      produced by the Mealy machine on the transition taken with input_t
    - Our Moore encoding: output comes from the state we're IN
    - Therefore: we must output AFTER the state update, not before

    Forward pass for each timestep:
        1. Form Kronecker product: μ_t = h_t ⊗ σ_t
        2. State update: h_{t+1} = A @ h_t + B @ μ_t
        3. Output (from NEW state): y_t = C @ h_{t+1}
    """

    def __init__(self, hoa_string: str, epsilon: float = 0.0):
        super().__init__()

        self.hoa = parse_hoa(hoa_string)
        self.moore = mealy_to_moore(self.hoa)

        self.num_states = self.moore.num_states
        self.num_inputs = self.moore.num_inputs
        self.num_output_aps = self.moore.num_output_aps
        self.epsilon = epsilon

        # Build matrices
        A = torch.eye(self.num_states)
        B = self._build_B()
        C = self._build_C()

        # Add noise if requested (for training experiments)
        if epsilon > 0:
            A = A + epsilon * torch.randn_like(A) * (A == 0).float()
            B = B + epsilon * torch.randn_like(B) * (B == 0).float()
            C = C + epsilon * torch.randn_like(C) * (C == 0).float()

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)

        # Initial state (one-hot)
        h0 = torch.zeros(self.num_states)
        h0[self.moore.start_state] = 1.0
        self.register_buffer("h0", h0)

    def _build_B(self) -> torch.Tensor:
        """Build B matrix for state transitions.

        B ∈ ℝ^{N × (N*M)} where N = num_states, M = num_inputs

        Column (i*M + j) corresponds to (state i, input j).
        B[:, i*M + j] = e_{T(i,j)} - e_i

        This ensures: if h_t = e_i and σ_t = e_j, then
        μ_t = e_i ⊗ e_j has a 1 at position i*M + j
        B @ μ_t = e_{T(i,j)} - e_i
        h_{t+1} = A @ h_t + B @ μ_t = e_i + e_{T(i,j)} - e_i = e_{T(i,j)}
        """
        N = self.num_states
        M = self.num_inputs
        B = torch.zeros(N, N * M)

        for state in range(N):
            for input_idx in range(M):
                col = state * M + input_idx
                next_state = self.moore.transitions.get((state, input_idx), state)

                # B[:, col] = e_{next_state} - e_{state}
                if next_state != state:
                    B[state, col] = -1.0
                    B[next_state, col] = 1.0
                # If next_state == state, column stays zero (no change needed)

        return B

    def _build_C(self) -> torch.Tensor:
        """Build C matrix for output mapping.

        C ∈ ℝ^{num_outputs × N} C[k, state] = 1 if output AP k is true in moore state
        'state'
        """
        C = torch.zeros(self.num_output_aps, self.num_states)

        for state, output_vals in enumerate(self.moore.state_outputs):
            for k, val in enumerate(output_vals):
                C[k, state] = 1.0 if val else 0.0

        return C

    def forward(self, inputs: torch.Tensor, use_tanh: bool = False) -> torch.Tensor:
        """Forward pass through the SSM.

        CRITICAL: Output is computed AFTER state update to match Mealy semantics.

        Args:
            inputs: (batch, seq_len, num_inputs) one-hot encoded input symbols
            use_tanh: If True, apply tanh nonlinearity (for gradient-based training)

        Returns:
            outputs: (batch, seq_len, num_output_aps)
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        print(device)

        # Initialize state
        h = self.h0.unsqueeze(0).expand(batch_size, -1).clone()

        outputs = []
        for t in range(seq_len):
            sigma_t = inputs[:, t]  # (batch, num_inputs)

            # Kronecker product: μ_t = h_t ⊗ σ_t
            # Result shape: (batch, num_states * num_inputs)
            mu_t = torch.einsum("bi,bj->bij", h, sigma_t)
            mu_t = mu_t.reshape(batch_size, -1)

            # State update: h_{t+1} = A @ h_t + B @ μ_t
            h_new = h @ self.A.T + mu_t @ self.B.T

            if use_tanh:
                h = torch.tanh(h_new)
            else:
                h = h_new

            # Output from NEW state (after transition)
            # This matches Mealy semantics: output_t is produced on transition at time t
            y_t = h @ self.C.T
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def simulate_mealy(self, input_sequence: List[int]) -> List[List[bool]]:
        """Simulate the original Mealy machine directly.

        Returns outputs produced on each transition.
        """
        mealy_state = self.hoa.start_state
        outputs = []

        for input_idx in input_sequence:
            key = (mealy_state, input_idx)
            if key in self.moore.mealy_transitions:
                next_state, output_tuple = self.moore.mealy_transitions[key]
                outputs.append(list(output_tuple))
                mealy_state = next_state
            else:
                # No transition defined - shouldn't happen for complete automata
                outputs.append([False] * self.num_output_aps)

        return outputs

    def simulate_moore(self, input_sequence: List[int]) -> List[List[bool]]:
        """Simulate the Moore machine with output AFTER transition.

        This should match simulate_mealy() exactly.
        """
        moore_state = self.moore.start_state
        outputs = []

        for input_idx in input_sequence:
            # Transition first
            next_state = self.moore.transitions.get(
                (moore_state, input_idx), moore_state
            )
            moore_state = next_state

            # Then output from new state
            output = self.moore.state_outputs[moore_state]
            outputs.append(output)

        return outputs

    def verify(self, input_sequence: List[int], verbose: bool = True) -> bool:
        """Verify SSM matches discrete automaton."""
        # Get discrete outputs (Mealy simulation)
        mealy_outputs = self.simulate_mealy(input_sequence)
        moore_outputs = self.simulate_moore(input_sequence)

        # Get SSM outputs
        inputs_tensor = torch.zeros(1, len(input_sequence), self.num_inputs)
        for t, idx in enumerate(input_sequence):
            inputs_tensor[0, t, idx] = 1.0

        with torch.no_grad():
            ssm_outputs = self(inputs_tensor, use_tanh=False)

        all_match = True
        for t in range(len(input_sequence)):
            mealy_out = mealy_outputs[t]
            moore_out = moore_outputs[t]
            ssm_out = ssm_outputs[0, t].tolist()
            ssm_binary = [v > 0.5 for v in ssm_out]

            mealy_match = mealy_out == ssm_binary
            all_match = all_match and mealy_match

            if verbose:
                status = "✓" if mealy_match else "✗"
                print(
                    f"t={t}: input={input_sequence[t]:2d} | "
                    f"mealy={mealy_out} | moore={moore_out} | ssm={ssm_binary} | "
                    f"raw={[f'{v:.2f}' for v in ssm_out]} {status}"
                )

        return all_match

    def print_info(self):
        """Print model information."""
        print("=" * 70)
        print("HOA SSM (Mealy-to-Moore with Output After Transition)")
        print("=" * 70)
        print(f"Original Mealy states: {self.hoa.num_states}")
        print(f"Expanded Moore states: {self.num_states}")
        print(
            f"Input symbols: {self.num_inputs} (= 2^{len(self.moore.input_ap_indices)})"
        )
        print(f"Output APs: {self.num_output_aps}")
        print(f"Start state: {self.moore.start_state}")

        input_names = [self.hoa.ap_names[i] for i in self.moore.input_ap_indices]
        output_names = [self.hoa.ap_names[i] for i in self.moore.output_ap_indices]
        print(f"\nInput APs (sorted): {input_names}")
        print(f"Output APs (sorted): {output_names}")

        print("\nMoore states:")
        for i, (mealy_state, output) in self.moore.moore_to_mealy.items():
            print(f"  {i}: Mealy state {mealy_state}, output {list(output)}")

        print("Matrix shapes:")
        print(f"  A: {tuple(self.A.shape)} (identity)")
        print(f"  B: {tuple(self.B.shape)} (transition encoding)")
        print(f"  C: {tuple(self.C.shape)} (output encoding)")

        print("\nC matrix (output mapping):")
        print(self.C.data)


# =============================================================================
# DEFAULT HOA FOR TESTING
# =============================================================================

ARBITER_HOA = """HOA: v1
States: 5
Start: 0
AP: 6 "g_0" "c_0" "r_0" "g_1" "c_1" "r_1"
acc-name: all
Acceptance: 0 t
properties: trans-labels explicit-labels state-acc deterministic
controllable-AP: 0 3
--BODY--
State: 0
[!0&1&!3&4 | !0&1&!3&!5 | !0&!2&!3&4 | !0&!2&!3&!5] 0
[!0&1&!3&!4&5 | !0&!2&!3&!4&5] 1
[!0&!1&2&!3&4 | !0&!1&2&!3&!5] 2
[0&!1&2&!3&!4&5] 3
State: 1
[!0&1&!3&4 | !0&!2&!3&4] 0
[!0&!1&2&!3&4] 2
[!0&1&3&!4 | !0&!2&3&!4] 0
[!0&!1&2&3&!4] 4
State: 2
[!0&1&!3&4 | !0&1&!3&!5] 0
[!0&1&!3&!4&5] 1
[0&!1&!3&!4&5] 3
[0&!1&!3&4 | 0&!1&!3&!5] 0
State: 3
[!0&1&!3&4] 0
[!0&!1&2&!3&4] 2
[!0&1&3&!4 | !0&!2&3&!4] 0
[!0&!1&2&3&!4] 4
[0&!1&!2&!3&4] 0
State: 4
[!0&1&!3&4 | !0&1&!3&!5] 0
[!0&1&!3&!4&5] 1
[!0&!1&!3&4 | !0&!1&!3&!5] 2
[!0&!1&3&!4&5] 2
--END--"""


if __name__ == "__main__":
    print("Creating HOA_SSM...\n")
    model = HOA_SSM(ARBITER_HOA, epsilon=0.0)
    model.print_info()

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Test sequence
    # Input encoding: bits are [c_0, r_0, c_1, r_1] in positions [3,2,1,0]
    # c_0=0, r_0=1, c_1=0, r_1=1 -> 0b0101 = 5
    print("\nTest: r_0=1, r_1=1, c_0=0, c_1=0 (both request, neither critical)")
    print("Input bits: c_0=0, r_0=1, c_1=0, r_1=1 -> index 5")
    test_input = [5, 5, 5, 5, 5]
    model.verify(test_input)

    # Another test: only client 0 requests
    print("\nTest: r_0=1, r_1=0 (only client 0 requests)")
    # c_0=0, r_0=1, c_1=0, r_1=0 -> 0b0100 = 4
    test_input2 = [4, 4, 4, 4, 4]
    model.verify(test_input2)
    print(model.A)
