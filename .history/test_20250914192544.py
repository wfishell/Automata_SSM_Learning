#!/usr/bin/env python3
"""
Convert binary traces into a prefix-closed dataset for passive RPNI learning of a Mealy machine.

USAGE:
    python convert_binary_traces.py traces.txt "go,req,cancel" "grant"

Each line of traces.txt should look like:
    1,1,1,0;1,0,0,1;0,1,0,0;

Where the first |inputs| bits are inputs, and the remaining |outputs| bits are outputs.
"""

import argparse
import sys


def parse_binary_trace(line: str, input_names, output_names):
    """
    Convert one raw binary trace line into a list of (input, output) pairs.
    """
    steps = [s.strip() for s in line.strip().split(";") if s.strip()]
    trace = []
    for idx, step in enumerate(steps):
        bits = [b.strip() for b in step.split(",")]
        if len(bits) != len(input_names) + len(output_names):
            raise ValueError(
                f"Step {idx}: got {len(bits)} bits, expected {len(input_names) + len(output_names)}. Step={step}"
            )
        in_bits = bits[: len(input_names)]
        out_bits = bits[len(input_names) :]

        input_tok = ",".join(f"{name}={val}" for name, val in zip(input_names, in_bits))
        output_tok = ",".join(
            f"{name}={val}" for name, val in zip(output_names, out_bits)
        )

        trace.append((input_tok, output_tok))
    return trace


def make_prefix_closed_dataset(lines, input_names, output_names):
    dataset = []
    for line in lines:
        trace = parse_binary_trace(line, input_names, output_names)
        # add all prefixes
        for k in range(1, len(trace) + 1):
            dataset.append(trace[:k])
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary traces to prefix-closed dataset for RPNI Mealy learning."
    )
    parser.add_argument(
        "trace_file", help="Path to file with binary traces (one per line)."
    )
    parser.add_argument(
        "inputs", help="Comma-separated input AP names, e.g. 'go,req,cancel'"
    )
    parser.add_argument("outputs", help="Comma-separated output AP names, e.g. 'grant'")
    args = parser.parse_args()

    input_names = [s.strip() for s in args.inputs.split(",") if s.strip()]
    output_names = [s.strip() for s in args.outputs.split(",") if s.strip()]
    if not input_names or not output_names:
        print("Error: both inputs and outputs must be non-empty.", file=sys.stderr)
        sys.exit(1)

    with open(args.trace_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = make_prefix_closed_dataset(lines, input_names, output_names)

    print(f"Loaded {len(lines)} traces")
    print(f"Created dataset with {len(dataset)} prefix-closed sequences")
    print("Sample (first 3 entries):")
    for d in dataset[:3]:
        print(d)


if __name__ == "__main__":
    main()
