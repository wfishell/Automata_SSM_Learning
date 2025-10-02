# check_trace.py
import re
import sys
import subprocess

if len(sys.argv) < 3:
    print("Usage: python check_trace.py <hoa_file> <ap1> <ap2> ...")
    sys.exit(1)

hoa_file = sys.argv[1]
aps = sys.argv[2:]

trace = []

with open("/Users/will/github/corp/examples/simple3/hoax_raw.txt", "r") as f:
    for line in f:
        # Look for {...}
        start = line.find("{")
        end = line.find("}")
        if start == -1 or end == -1:
            continue
        raw = line[start+1:end].strip()
        if raw:
            # Handle things like "'a', 'p2'" → [a, p2]
            present = {tok.strip().strip("'\"") for tok in raw.split(",")}
        else:
            present = set()

        assignment = []
        for ap in aps:
            if ap in present:
                assignment.append(ap)
            else:
                assignment.append("!" + ap)
        trace.append("&".join(assignment))

# Build Spot word
spot_word = ";".join(trace) + ";cycle{1}"

# Save to file
with open("/Users/will/github/corp/examples/simple3/trace.txt", "w") as f:
    f.write(spot_word + "\n")

print("📄 Spot trace written to spot_trace.txt")

# Run autfilt
print("🔍 Checking trace with autfilt...")
try:
    result = subprocess.run(
        ["autfilt", hoa_file, f"--accept-word={spot_word}"],
        check=False,
        text=True,
        capture_output=True
    )
    if result.returncode == 0:
        print("✅ Trace is ACCEPTED by automaton.")
    else:
        print("❌ Trace is REJECTED by automaton.")
except FileNotFoundError:
    print("Error: autfilt not found in PATH. Install Spot and make sure autfilt is accessible.")