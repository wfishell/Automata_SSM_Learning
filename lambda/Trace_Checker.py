#!/usr/bin/env python3
import subprocess
import sys


def check_trace(hoa_file, trace):
    """Return True if trace is accepted by HOA automaton, else False."""
    try:
        result = subprocess.run(
            ["autfilt", hoa_file, f"--accept-word={trace}"],
            capture_output=True,
            text=True,
        )
        # autfilt exits with 0 if accepted, 1 if rejected
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Failed to check trace {trace}: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python Trace_Checker.py <hoa_file> <trace_file>")
        sys.exit(1)

    hoa_file = sys.argv[1]
    trace_file = sys.argv[2]

    with open(trace_file, "r") as f:
        traces = [
            line.strip() for line in f if line.strip() and not line.startswith("------")
        ]

    total = len(traces)
    accepted = 0

    for i, trace in enumerate(traces, 1):
        is_ok = check_trace(hoa_file, trace)
        if is_ok:
            accepted += 1

    print("\n=== Summary ===")
    print(f"Total traces: {total}")
    print(f"Accepted:     {accepted}")
    print(f"Rejected:     {total - accepted}")
    print(f"Acceptance %: {accepted/total*100:.2f}%")


if __name__ == "__main__":
    main()
