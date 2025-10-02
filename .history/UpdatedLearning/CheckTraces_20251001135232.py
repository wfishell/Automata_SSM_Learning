# check_traces_simple.py
import sys
import subprocess

if len(sys.argv) != 3:
    print("Usage: python check_traces_simple.py <hoa_file> <trace_file>")
    sys.exit(1)

hoa_file = sys.argv[1]
trace_file = sys.argv[2]

total = 0
accepted = 0

with open(trace_file, "r") as f:
    for line in f:
        trace = line.strip()
        if not trace:
            continue
        total += 1
        result = subprocess.run(
            ["autfilt", hoa_file, f"--accept-word={trace}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            accepted += 1

if total == 0:
    print("No traces found in file.")
else:
    percent = (accepted / total) * 100
    print(f"✅ {accepted}/{total} traces accepted ({percent:.2f}%).")
