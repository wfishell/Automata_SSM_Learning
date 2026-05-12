"""Parity Flip Transformer.

Takes a trace file and flips output bits when the cumulative input parity is odd.
"""

import sys
from pathlib import Path
from typing import Dict, List


def parse_step(step: str) -> Dict[str, int]:
    """Parse a single step like 'g_0&!r_0&g_1' into {var: value}."""
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1
    return valuation


def step_to_string(valuation: Dict[str, int], var_order: List[str]) -> str:
    """Convert valuation dict back to string like '!g_0&r_0&!g_1'."""
    parts = []
    for var in var_order:
        if valuation[var] == 0:
            parts.append(f"!{var}")
        else:
            parts.append(var)
    return "&".join(parts)


def apply_parity_flip(
    trace_file: str, inputs: List[str], outputs: List[str], output_file: str = None
):
    """Apply parity flip transformation to traces."""

    lines = Path(trace_file).read_text().splitlines()

    if output_file is None:
        output_file = str(Path(trace_file).with_suffix(".parity.txt"))

    all_vars = inputs + outputs
    transformed_lines = []

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Handle cycle notation
        cycle_suffix = ""
        if "cycle{" in line:
            cycle_suffix = "cycle{" + line.split("cycle{")[1]
            line = line.split("cycle{")[0].rstrip(";")

        steps = [s for s in line.split(";") if s]

        phi = 0  # cumulative input count
        transformed_steps = []

        for step in steps:
            valuation = parse_step(step)

            # Count input bits and update phi
            input_sum = sum(valuation.get(inp, 0) for inp in inputs)
            phi += input_sum
            parity = phi % 2

            # Flip outputs if parity is odd
            if parity == 1:
                for out in outputs:
                    if out in valuation:
                        valuation[out] = 1 - valuation[out]

            # Convert back to string
            transformed_steps.append(step_to_string(valuation, all_vars))

        transformed_line = ";".join(transformed_steps)
        if cycle_suffix:
            transformed_line += ";" + cycle_suffix

        transformed_lines.append(transformed_line)

        # Print comparison for first trace
        if line_idx == 0:
            print("=== First trace comparison ===\n")
            original_steps = [s for s in line.split(";") if s]
            phi_check = 0
            for t, (orig, trans) in enumerate(zip(original_steps, transformed_steps)):
                orig_val = parse_step(orig)
                input_sum = sum(orig_val.get(inp, 0) for inp in inputs)
                phi_check += input_sum
                parity = phi_check % 2

                orig_outs = [orig_val.get(o, 0) for o in outputs]
                trans_val = parse_step(trans)
                trans_outs = [trans_val.get(o, 0) for o in outputs]

                print(f"t={t}: inputs={input_sum}, phi={phi_check}, parity={parity}")
                print(f"      original outputs:    {orig_outs}")
                print(f"      transformed outputs: {trans_outs}")
                print()

    # Write output
    Path(output_file).write_text("\n".join(transformed_lines))
    print(f"[+] Wrote transformed traces to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python parity_flip.py <trace_file> <inputs> <outputs>")
        print(
            "  Example: python parity_flip.py traces.txt r_0,r_1 g_0,g_1,a_0,b_0,c_0,a_1,b_1,c_1"
        )
        sys.exit(1)

    trace_file = sys.argv[1]
    inputs = sys.argv[2].split(",")
    outputs = sys.argv[3].split(",")

    apply_parity_flip(trace_file, inputs, outputs)
