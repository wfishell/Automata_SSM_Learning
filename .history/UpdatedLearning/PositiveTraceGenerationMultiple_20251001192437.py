import sys
from pathlib import Path

# Import from your new simplified pipeline
from PositiveTraceGeneration import trace_from_hoa


def generate_positive_traces(hoa_file: str, config_file: str, num_traces: int, output_file: str):
    traces = []

    for i in range(num_traces):
        print(f"[+] Run {i+1}/{num_traces}")
        pos_trace = trace_from_hoa(hoa_file, config_file)
        print(f"    Positive Trace {i+1}: GENERATED ✅")
        traces.append(pos_trace)

    # Write output file (just positives, no divider since no negatives now)
    Path(output_file).write_text("\n".join(traces))
    print(f"[+] Wrote {num_traces} positive traces to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: batch_positive_traces.py <system.hoa> <config.toml> <N> <output.txt>")
        sys.exit(1)

    hoa_file = sys.argv[1]
    config_file = sys.argv[2]
    num_traces = int(sys.argv[3])
    output_file = sys.argv[4]

    generate_positive_traces(hoa_file, config_file, num_traces, output_file)
