"""
Trace Converter for RPNI
Reads traces from a text file, strips cycle markers,
splits into I/O valuations, and runs convert_i_o_traces_for_RPNI.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# TODO: import your actual conversion function
# from your_module import convert_i_o_traces_for_RPNI


def parse_step(
    step: str, inputs: List[str], outputs: List[str]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Parse one step like 'cancel&!grant&!go&!req' into input/output dicts."""
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1
    # Split into inputs and outputs
    in_dict = {k: valuation.get(k, 0) for k in inputs}
    out_dict = {k: valuation.get(k, 0) for k in outputs}
    return in_dict, out_dict


def parse_trace(line: str, inputs: List[str], outputs: List[str]):
    """Parse one full trace line into [(inputs, outputs), ...]."""
    line = line.strip()
    # Remove cycle{…} if present
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = line.split(";")
    trace = [parse_step(s, inputs, outputs) for s in steps if s]
    return trace


def process_file(trace_file: str, inputs: List[str], outputs: List[str]):
    lines = Path(trace_file).read_text().splitlines()
    for i, line in enumerate(lines):
        trace = parse_trace(line, inputs, outputs)
        print(f"\n[Trace {i+1}] Raw parsed trace:")
        print(trace)
        # Call conversion
        try:
            converted = convert_i_o_traces_for_RPNI(trace)
        except NameError:
            converted = f"(Dummy output) would convert {trace}"
        print("[Trace {i+1}] Converted for RPNI:")
        print(converted)


if __name__ == "__main__":
    # Example: define which variables are inputs and outputs
    INPUTS = ["cancel", "req", "go"]
    OUTPUTS = ["grant"]

    process_file("TraceGeneration_UPDATEPATHS/Result.txt", INPUTS, OUTPUTS)
