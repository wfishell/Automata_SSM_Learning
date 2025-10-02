"""
Batch Trace Generator
Authors: Nikolaus Holzer, Will Fishell
Date: September 2025

Runs pipeline.py multiple times to generate Spot traces and writes them to file.
"""

import subprocess
import sys
from pathlib import Path


def run_pipeline(tlsf_file: str, config_file: str):
    """Run pipeline.py and capture its JSON-like stdout (trace info)."""
    cmd = [sys.executable, "pipeline.py", tlsf_file, config_file]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def extract_trace(stdout: str):
    """Pull the trace string from pipeline.py's output log/dict."""
    import re

    # Match something like "'trace': '...';"
    m = re.search(r"'trace':\s*'([^']+)'", stdout)
    if not m:
        raise ValueError("Trace not found in pipeline.py output")
    return m.group(1)


def generate_traces(
    tlsf_file: str, config_file: str, num_traces: int, output_file: str
):
    traces = []
    for i in range(num_traces):
        print(f"[+] Run {i+1}/{num_traces}")
        stdout = run_pipeline(tlsf_file, config_file)
        trace = extract_trace(stdout)
        traces.append(trace)

    Path(output_file).write_text("\n".join(traces))
    print(f"[+] Wrote {num_traces} traces to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: batch_traces.py <spec.tlsf> <config.toml> <N> <output.txt>")
        sys.exit(1)

    tlsf_file = sys.argv[1]
    config_file = sys.argv[2]
    num_traces = int(sys.argv[3])
    output_file = sys.argv[4]

    generate_traces(tlsf_file, config_file, num_traces, output_file)
