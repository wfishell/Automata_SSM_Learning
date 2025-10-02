"""
Batch Trace Generator
Authors: Nikolaus Holzer, Will Fishell
Date: September 2025

Runs pipeline.py multiple times to generate Spot traces and writes them to file.
"""

import sys
from pathlib import Path

# Import pipeline function directly
from pipeline import pipeline


def generate_traces(tlsf_file: str, config_file: str, num_traces: int, output_file: str):
    traces = []
    for i in range(num_traces):
        print(f"[+] Run {i+1}/{num_traces}")
        result = pipeline(tlsf_file, config_file)  # directly call function
        trace = result["trace"]
        accepted = result["accepted"]

        # Print acceptance status for each run
        status = "ACCEPTED" if accepted else "REJECTED"
        print(f"    Trace {i+1} -> {status}")

        # Store both trace and acceptance info in output file
        traces.append(f"{trace}    # {status}")

    Path(output_file).write_text("\n".join(traces))
    print(f"[+] Wrote {num_traces} traces with acceptance status to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: batch_traces.py <spec.tlsf> <config.toml> <N> <output.txt>")
        sys.exit(1)

    tlsf_file = sys.argv[1]
    config_file = sys.argv[2]
    num_traces = int(sys.argv[3])
    output_file = sys.argv[4]

    generate_traces(tlsf_file, config_file, num_traces, output_file)
