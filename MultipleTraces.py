#!/usr/bin/env python3
import argparse
import subprocess
import sys

USAGE = (
    "Generate multiple parsed IO traces from FiniteTraceGenerator.py output.\n"
    "Example:\n"
    "  python MultipleTraces.py \\\n"
    "    --controller learned_mealy.dot \\\n"
    "    --inputs r,c \\\n"
    "    --outputs g0,g1 \\\n"
    "    --steps 20 \\\n"
    "    --assumption 'G (!c | X (g0 | X g0))' \\\n"
    "    --num 12 \\\n"
    "    --out TestTraces.txt\n"
)

def parse_single_trace(raw: str) -> str:
    """
    Convert generator output like:
      # inputs
      a
      0
      1
      ...
      # outputs
      b,c
      0,0
      1,1
      ...
    into: '0,0,0;1,1,1;...;' (one line per trace)
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    try:
        i_inputs = lines.index("# inputs")
        i_outputs = lines.index("# outputs")
    except ValueError:
        raise ValueError("Could not find '# inputs' / '# outputs' sections in generator output.")

    if i_inputs + 1 >= len(lines) or lines[i_inputs + 1] == "# outputs":
        raise ValueError("Missing input header (e.g., 'a').")
    if i_outputs + 1 >= len(lines):
        raise ValueError("Missing output header (e.g., 'b,c').")

    # Values after the header lines
    inputs = lines[i_inputs + 2 : i_outputs]
    outputs = lines[i_outputs + 2 : ]

    if len(inputs) != len(outputs):
        raise ValueError(f"Inputs/outputs length mismatch: {len(inputs)} vs {len(outputs)}")

    steps = [f"{inp},{out}" for inp, out in zip(inputs, outputs)]
    return ";".join(steps) + ";"  # trailing ';' as requested

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run FiniteTraceGenerator multiple times and parse traces into single-line format.",
        epilog=USAGE,
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--controller", required=True, help="Path to controller .dot file.")
    p.add_argument("--inputs", required=True,
                   help="Comma-separated input APs, e.g. 'r,c' (no spaces).")
    p.add_argument("--outputs", required=True,
                   help="Comma-separated output APs, e.g. 'g0,g1' (no spaces).")
    p.add_argument("--steps", type=int, required=True,
                   help="Number of random steps per generated trace.")
    p.add_argument("--assumption", default="",
                   help="LTL assumption string (quote it in the shell). Default: empty.")
    p.add_argument("--num", type=int, default=1,
                   help="Number of traces (seeds) to generate. Default: 1.")
    p.add_argument("--out", default="parsed_traces.txt",
                   help="Output file name for parsed traces. Default: parsed_traces.txt")
    p.add_argument("--generator", default="FiniteTraceGenerator.py",
                   help="Generator script path/name if not in CWD. Default: FiniteTraceGenerator.py")
    return p

def main():
    args = build_arg_parser().parse_args()

    # fresh file
    open(args.out, "w").close()

    for seed in range(args.num):
        cmd = [
            sys.executable, args.generator,
            args.controller,
            "--inputs", args.inputs,
            "--outputs", args.outputs,
            "--rand-steps", str(args.steps),
            "--assume", args.assumption,
            "--seed", str(seed),
        ]

        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[Seed {seed}] Generator failed with return code {e.returncode}")
            print("STDOUT:\n" + e.stdout)
            print("STDERR:\n" + e.stderr)
            sys.exit(1)

        try:
            line = parse_single_trace(result.stdout)
        except Exception as e:
            print(f"[Seed {seed}] Parse error: {e}")
            print("Generator stdout was:\n" + result.stdout)
            sys.exit(2)

        with open(args.out, "a") as f:
            f.write(line + "\n")

        print(f"[Seed {seed}] appended to {args.out}: {line}")

if __name__ == "__main__":
    main()