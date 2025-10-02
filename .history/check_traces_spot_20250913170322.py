# check_trace.py
import re
import sys
import subprocess

if len(sys.argv) < 3:
    print("Usage: python check_trace.py <hoa_file> <ap1> <ap2> ...")
    sys.exit(1)

hoa_file = sys.argv[1]
aps = sys.argv[2:]

# Regex to capture the set {...}
pattern = re.compile(r"\{([^}]*)\}")
trace = []

with open("hoax_raw.txt", "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue
        raw_aps = match.group(1).strip()
        if raw_aps:
            present = {ap.strip() for ap in raw_aps.split(",")}
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
