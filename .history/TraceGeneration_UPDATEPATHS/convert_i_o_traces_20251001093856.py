"""
Trace Converter for RPNI
Reads traces from a text file, strips cycle markers,
splits into I/O valuations, and runs convert_i_o_traces_for_RPNI.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from aalpy.utils import convert_i_o_traces_for_RPNI


def parse_step(step: str, inputs: List[str], outputs: List[str]) -> Tuple[Dict[str,int], str]:
    """Parse one step like 'cancel&!grant&!go&!req' into (input_dict, output_symbol)."""
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1

    # Inputs remain a dict
    in_dict = {k: valuation.get(k, 0) for k in inputs}

    # Outputs collapsed into a single symbol (string of 0/1)
    out_symbol = "".join(str(valuation.get(k, 0)) for k in outputs)

    return in_dict, out_symbol


def parse_trace(line: str, inputs: List[str], outputs: List[str]):
    """Parse one full trace line into [(inputs, output_symbol), ...]."""
    line = line.strip()
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = [s for s in line.split(";") if s]  # drop empty pieces

    trace = []
    for s in steps:
        parsed = parse_step(s, inputs, outputs)
        if not isinstance(parsed, tuple) or len(parsed) != 2:
            raise ValueError(f"Malformed step: {parsed} from string '{s}'")
        trace.append(parsed)

    return trace



def process_file(trace_file: str, inputs: List[str], outputs: List[str]):
    lines = Path(trace_file).read_text().splitlines()
    for i, line in enumerate(lines):
        trace = parse_trace(line, inputs, outputs)
        print(f"\n[Trace {i+1}] Raw parsed trace:")
        print(trace)
        try:
            converted = convert_i_o_traces_for_RPNI(trace)

        print(f"[Trace {i+1}] Converted for RPNI:")
        print(converted)


if __name__ == "__main__":
    INPUTS = ["cancel", "req", "go"]
    OUTPUTS = ["grant"]

    process_file("TraceGeneration_UPDATEPATHS/Result.txt", INPUTS, OUTPUTS)
