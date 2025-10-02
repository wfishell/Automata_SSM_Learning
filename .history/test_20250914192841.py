#!/usr/bin/env python3
"""
Learn a deterministic Mealy machine from binary traces using passive RPNI.

USAGE:
    python learn_mealy.py traces.txt "go,req,cancel" "grant" --dump-hoa result.hoa

Each line of traces.txt should look like:
    1,1,1,0;1,0,0,1;0,1,0,0;

Where the first |inputs| bits are inputs, and the remaining |outputs| bits are outputs.
"""

import argparse
import sys

# --- Import run_RPNI from AALPy ---
try:
    from aalpy.learning_algs.deterministic_passive.RPNI import run_RPNI
except ImportError:
    print(
        "Error: Could not import AALPy. Install with `pip install aalpy`.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_binary_trace(line: str, input_names, output_names):
    """Convert one raw binary trace line into a list of (input, output) pairs."""
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


def make_io_dataset(lines, input_names, output_names):
    """
    Return dataset as list of (input_sequence, output_sequence) pairs.
    """
    dataset = []
    for line in lines:
        trace = parse_binary_trace(line, input_names, output_names)
        in_seq = [inp for inp, _ in trace]
        out_seq = [out for _, out in trace]
        dataset.append((in_seq, out_seq))
    return dataset


def save_hoa(model, path: str, input_names, output_names):
    """Save the learned Mealy machine in HOA-like format (for Spot compatibility)."""
    with open(path, "w") as f:
        f.write("HOA: v1\n")
        f.write('name: "Learned Mealy Machine"\n')
        f.write('tool: "RPNI Learning"\n')
        f.write(f"States: {len(model.states)}\n")

        aps = input_names + output_names
        f.write(f"AP: {len(aps)}")
        for ap in aps:
            f.write(f' "{ap}"')
        f.write("\n")

        # outputs are controllable
        out_idx = [str(i) for i, ap in enumerate(aps) if ap in output_names]
        f.write("controllable-AP: " + " ".join(out_idx) + "\n")

        f.write(f"Start: {model.states.index(model.initial_state)}\n")
        f.write("acc-name: all\n")
        f.write("Acceptance: 0 t\n")
        f.write("properties: deterministic\n")
        f.write("--BODY--\n")

        state_to_id = {s: i for i, s in enumerate(model.states)}

        for state_id, state in enumerate(model.states):
            f.write(f"State: {state_id}\n")
            for inp, target in state.transitions.items():
                target_id = state_to_id[target]
                out_val = state.output_fun[inp]

                in_bits = [int(p.split("=")[1]) for p in inp.split(",")]
                out_bits = [int(p.split("=")[1]) for p in out_val.split(",")]
                all_bits = in_bits + out_bits

                ap_cond = []
                for i, b in enumerate(all_bits):
                    ap_cond.append(str(i) if b == 1 else f"!{i}")
                cond_str = "&".join(ap_cond) if ap_cond else "t"

                f.write(f"[{cond_str}] {target_id}\n")

        f.write("--END--\n")

    print(f"HOA saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Learn a deterministic Mealy machine from binary traces via RPNI."
    )
    parser.add_argument(
        "trace_file", help="Path to file with binary traces (one per line)."
    )
    parser.add_argument(
        "inputs", help="Comma-separated input AP names, e.g. 'go,req,cancel'"
    )
    parser.add_argument("outputs", help="Comma-separated output AP names, e.g. 'grant'")
    parser.add_argument(
        "--dump-hoa",
        default="result.hoa",
        help="Path to save learned HOA (default=result.hoa)",
    )
    args = parser.parse_args()

    input_names = [s.strip() for s in args.inputs.split(",") if s.strip()]
    output_names = [s.strip() for s in args.outputs.split(",") if s.strip()]
    if not input_names or not output_names:
        print("Error: both inputs and outputs must be non-empty.", file=sys.stderr)
        sys.exit(1)

    with open(args.trace_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    dataset = make_prefix_closed_dataset(lines, input_names, output_names)

    print(
        f"Loaded {len(lines)} traces, created {len(dataset)} prefix-closed sequences."
    )

    # Learn Mealy machine
    mealy = run_RPNI(data=dataset, automaton_type="mealy", algorithm="classic")
    print(f"Learned Mealy machine with {len(mealy.states)} states.")

    save_hoa(mealy, args.dump_hoa, input_names, output_names)


if __name__ == "__main__":
    main()
