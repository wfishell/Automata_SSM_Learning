

import sys
from pathlib import Path

# Import pipeline function directly
from pipeline import pipeline


def generate_traces(tlsf_file: str, config_file: str, num_traces: int, output_file: str):
    lines = []
    for i in range(num_traces):
        print(f"[+] Run {i+1}/{num_traces}")
        result = pipeline(tlsf_file, config_file)

        trace = result["trace"]
        accepted = result["accepted"]
        status = "ACCEPTED ✅" if accepted else "REJECTED ❌"

        # Explicit console message
        print(f"    Trace {i+1} status: {status}")

        # Save only trace (no status) to output file
        lines.append(trace)

    Path(output_file).write_text("\n".join(lines))
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
