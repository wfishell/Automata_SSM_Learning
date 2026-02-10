#!/usr/bin/env python3
"""Generate SPOT-format traces from HOA_SSM.

Imports the HOA_SSM module and uses its SSM forward pass (or Mealy simulation)
to generate traces in SPOT format.

Usage:
    python trace_hoa_generation.py --hoa arbiter.hoa --num-traces 100 --length 30 --output traces.txt
    python trace_hoa_generation.py --num-traces 50 --length 20  # Uses default arbiter
    python trace_hoa_generation.py --no-ssm  # Use dictionary lookup instead of SSM
"""

import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch

from HOA_SSM import ARBITER_HOA, HOA_SSM

# =============================================================================
# TRACE FORMATTING
# =============================================================================


def input_idx_to_ap_values(
    input_idx: int, input_ap_indices: List[int]
) -> Dict[int, bool]:
    """Convert input index to AP boolean values.

    Uses MSB-first encoding to match HOA_SSM.
    """
    num_input_aps = len(input_ap_indices)
    sorted_aps = sorted(input_ap_indices)
    values = {}

    for i, ap_idx in enumerate(sorted_aps):
        bit_pos = num_input_aps - 1 - i  # MSB first
        values[ap_idx] = bool((input_idx >> bit_pos) & 1)

    return values


def output_to_ap_values(
    output: List[bool], output_ap_indices: List[int]
) -> Dict[int, bool]:
    """Convert output list to AP boolean values."""
    sorted_aps = sorted(output_ap_indices)
    return {ap_idx: output[i] for i, ap_idx in enumerate(sorted_aps)}


def format_spot_step(ap_values: Dict[int, bool], ap_names: List[str]) -> str:
    """Format a single step as SPOT literal string.

    Example: "g_0&!c_0&r_0&!g_1&!c_1&r_1"
    """
    literals = []
    for ap_idx in sorted(ap_values.keys()):
        ap_name = ap_names[ap_idx]
        if ap_values[ap_idx]:
            literals.append(ap_name)
        else:
            literals.append(f"!{ap_name}")
    return "&".join(literals)


# =============================================================================
# TRACE GENERATION
# =============================================================================


def generate_random_input(num_input_aps: int) -> int:
    """Generate a random input symbol index."""
    return random.randint(0, (2**num_input_aps) - 1)


def generate_trace_from_ssm(
    model: HOA_SSM, input_sequence: List[int]
) -> List[List[bool]]:
    """Generate outputs using the actual SSM forward pass.

    This runs through the A, B, C matrices rather than dictionary lookup.
    """
    inputs_tensor = torch.zeros(1, len(input_sequence), model.num_inputs)
    for t, idx in enumerate(input_sequence):
        inputs_tensor[0, t, idx] = 1.0

    with torch.no_grad():
        y_hat = model(inputs_tensor, use_tanh=False)

    # Threshold to get binary outputs
    outputs = [[v > 0.5 for v in step] for step in y_hat[0].tolist()]
    return outputs


def generate_trace(
    model: HOA_SSM,
    length: int,
    input_sequence: Optional[List[int]] = None,
    use_ssm: bool = True,
) -> Tuple[str, List[int]]:
    """Generate a single trace of specified length.

    Args:
        model: The HOA_SSM model
        length: Number of steps in the trace
        input_sequence: Optional predetermined input sequence (random if None)
        use_ssm: If True, use SSM forward pass; if False, use dictionary lookup

    Returns:
        (spot_trace_string, input_sequence)
    """
    ap_names = model.hoa.ap_names
    input_ap_indices = model.moore.input_ap_indices
    output_ap_indices = model.moore.output_ap_indices
    num_input_aps = len(input_ap_indices)

    # Generate input sequence if not provided
    if input_sequence is None:
        input_sequence = [generate_random_input(num_input_aps) for _ in range(length)]

    # Generate outputs using SSM or dictionary
    if use_ssm:
        outputs = generate_trace_from_ssm(model, input_sequence)
    else:
        outputs = model.simulate_mealy(input_sequence)

    # Format as SPOT trace
    steps = []
    for t in range(len(input_sequence)):
        # Get input AP values
        input_vals = input_idx_to_ap_values(input_sequence[t], input_ap_indices)

        # Get output AP values
        output_vals = output_to_ap_values(outputs[t], output_ap_indices)

        # Combine all AP values
        all_ap_values = {**input_vals, **output_vals}

        # Format step
        step_str = format_spot_step(all_ap_values, ap_names)
        steps.append(step_str)

    # Join steps with semicolons
    trace_str = ";".join(steps)

    return trace_str, input_sequence


def generate_traces(
    model: HOA_SSM,
    num_traces: int,
    length: int,
    add_cycle: bool = True,
    variable_length: bool = False,
    min_length: int = 10,
    seed: Optional[int] = None,
    use_ssm: bool = True,
) -> List[str]:
    """Generate multiple traces.

    Args:
        model: The HOA_SSM model
        num_traces: Number of traces to generate
        length: Maximum length of each trace
        add_cycle: Whether to add cycle{1} annotation
        variable_length: If True, traces have random lengths
        min_length: Minimum length when variable_length is True
        seed: Random seed for reproducibility
        use_ssm: If True, use SSM forward pass; if False, use dictionary lookup

    Returns:
        List of SPOT-format trace strings
    """
    if seed is not None:
        random.seed(seed)

    traces = []
    for _ in range(num_traces):
        # Determine trace length
        if variable_length:
            trace_length = random.randint(min_length, length)
        else:
            trace_length = length

        # Generate trace
        trace_str, _ = generate_trace(model, trace_length, use_ssm=use_ssm)

        # Add cycle annotation if requested
        if add_cycle:
            trace_str += ";cycle{1}"

        traces.append(trace_str)

    return traces


def generate_exhaustive_traces(
    model: HOA_SSM, max_depth: int, add_cycle: bool = False, use_ssm: bool = True
) -> List[str]:
    """Generate traces by exhaustive exploration up to max_depth.

    Generates all possible input sequences up to the given depth.
    """
    num_inputs = model.num_inputs
    traces = []

    def dfs(input_seq: List[int], depth: int):
        if depth == max_depth:
            trace_str, _ = generate_trace(
                model, depth, input_sequence=input_seq, use_ssm=use_ssm
            )
            if add_cycle:
                trace_str += ";cycle{1}"
            traces.append(trace_str)
            return

        for input_idx in range(num_inputs):
            dfs(input_seq + [input_idx], depth + 1)

    dfs([], 0)
    return traces


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPOT-format traces from HOA_SSM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --num-traces 100 --length 30 --output traces.txt
  %(prog)s --hoa my_spec.hoa -n 50 -l 20 -o train.txt
  %(prog)s --exhaustive 4  # All traces of length 4
  %(prog)s --variable-length --min-length 10 --length 50
  %(prog)s --no-ssm  # Use dictionary lookup instead of SSM forward pass
        """,
    )
    parser.add_argument(
        "--hoa",
        type=str,
        default=None,
        help="HOA file (uses default arbiter if not provided)",
    )
    parser.add_argument(
        "--num-traces",
        "-n",
        type=int,
        default=100,
        help="Number of traces to generate (default: 100)",
    )
    parser.add_argument(
        "--length",
        "-l",
        type=int,
        default=30,
        help="Length of each trace (default: 30)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum length when using --variable-length (default: 10)",
    )
    parser.add_argument(
        "--variable-length",
        action="store_true",
        help="Generate traces with random lengths between min-length and length",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="traces.txt",
        help="Output file path (default: traces.txt)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-cycle", action="store_true", help="Don't add cycle annotations"
    )
    parser.add_argument(
        "--no-ssm",
        action="store_true",
        help="Use dictionary lookup instead of SSM forward pass",
    )
    parser.add_argument(
        "--exhaustive",
        type=int,
        default=None,
        help="Generate ALL traces up to given depth (overrides --num-traces)",
    )
    parser.add_argument(
        "--print-info", action="store_true", help="Print detailed model information"
    )
    parser.add_argument(
        "--print-sample",
        type=int,
        default=3,
        help="Number of sample traces to print (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SSM outputs match dictionary outputs",
    )
    args = parser.parse_args()

    # Load HOA
    if args.hoa:
        with open(args.hoa, "r") as f:
            hoa_string = f.read()
        print(f"Loaded HOA from {args.hoa}")
    else:
        hoa_string = ARBITER_HOA
        print("Using default arbiter HOA")

    # Create model
    model = HOA_SSM(hoa_string, epsilon=0.0)

    if args.print_info:
        model.print_info()

    # Print encoding info
    ap_names = model.hoa.ap_names
    input_names = [ap_names[i] for i in model.moore.input_ap_indices]
    output_names = [ap_names[i] for i in model.moore.output_ap_indices]

    use_ssm = not args.no_ssm

    print("\nModel info:")
    print(f"  Mealy states: {model.hoa.num_states}")
    print(f"  Moore states: {model.num_states}")
    print(f"  Input APs: {input_names}")
    print(f"  Output APs: {output_names}")
    print(f"  Input alphabet: {model.num_inputs} symbols")
    print(
        f"  Generation mode: {'SSM forward pass' if use_ssm else 'Dictionary lookup'}"
    )

    # Verification mode
    if args.verify:
        print("\n" + "=" * 70)
        print("VERIFICATION: Comparing SSM vs Dictionary outputs")
        print("=" * 70)

        num_test = 100
        length = 30
        mismatches = 0

        for i in range(num_test):
            input_seq = [
                generate_random_input(len(model.moore.input_ap_indices))
                for _ in range(length)
            ]

            ssm_outputs = generate_trace_from_ssm(model, input_seq)
            dict_outputs = model.simulate_mealy(input_seq)

            if ssm_outputs != dict_outputs:
                mismatches += 1
                if mismatches <= 3:  # Show first few mismatches
                    print(f"\nMismatch in trace {i}:")
                    for t in range(length):
                        if ssm_outputs[t] != dict_outputs[t]:
                            print(
                                f"  t={t}: input={input_seq[t]}, SSM={ssm_outputs[t]}, Dict={dict_outputs[t]}"
                            )

        if mismatches == 0:
            print(f"✓ All {num_test} traces match between SSM and dictionary!")
        else:
            print(f"✗ {mismatches}/{num_test} traces have mismatches")
        return

    # Generate traces
    if args.exhaustive is not None:
        expected = model.num_inputs**args.exhaustive
        print(f"\nGenerating exhaustive traces of depth {args.exhaustive}...")
        print(f"Expected: {expected} traces")

        if expected > 100000:
            response = input(f"This will generate {expected} traces. Continue? [y/N] ")
            if response.lower() != "y":
                print("Aborted.")
                return

        traces = generate_exhaustive_traces(
            model, args.exhaustive, add_cycle=not args.no_cycle, use_ssm=use_ssm
        )
    else:
        print(f"\nGenerating {args.num_traces} traces of length {args.length}...")
        traces = generate_traces(
            model,
            num_traces=args.num_traces,
            length=args.length,
            add_cycle=not args.no_cycle,
            variable_length=args.variable_length,
            min_length=args.min_length,
            seed=args.seed,
            use_ssm=use_ssm,
        )

    print(f"Generated {len(traces)} traces")

    # Print samples
    if args.print_sample > 0:
        print("\nSample traces:")
        for i, trace in enumerate(traces[: args.print_sample]):
            # Truncate for display
            if len(trace) > 150:
                display = trace[:150] + "..."
            else:
                display = trace
            print(f"  [{i}] {display}")

    # Save to file
    with open(args.output, "w") as f:
        for trace in traces:
            f.write(trace + "\n")

    print(f"\nTraces saved to {args.output}")


if __name__ == "__main__":
    main()
