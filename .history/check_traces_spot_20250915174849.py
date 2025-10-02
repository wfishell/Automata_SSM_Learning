#!/usr/bin/env python3
import os
import subprocess
import sys

if len(sys.argv) < 5:
    print(
        "Usage: python check_traces_spot.py <aps> <hoa_file> <hoa_traces_dir> <output_dir>"
    )
    print(
        "Example: python check_traces_spot.py 'a b c d' controller.hoa HOA_Traces SpotTraces"
    )
    sys.exit(1)
hoa_file = sys.argv[2]
trace_dir = sys.argv[3]  # e.g., HOA_Traces
output_dir = sys.argv[4]  # e.g., SpotTraces

aps_arg = sys.argv[1]

# Parse APs (split by whitespace, remove empty tokens)
aps = [ap.strip() for ap in aps_arg.split() if ap.strip()]
print("🔎 APs being used:", aps)

os.makedirs(output_dir, exist_ok=True)

valid_count = 0
total_count = 0

for fname in os.listdir(trace_dir):
    if not fname.endswith(".txt"):
        continue

    trace_path = os.path.join(trace_dir, fname)
    trace = []

    with open(trace_path, "r") as f:
        for line in f:
            start = line.find("{")
            end = line.find("}")
            if start == -1 or end == -1:
                continue
            raw = line[start + 1 : end].strip()
            if raw:
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

    if not trace:
        print(f"⚠️ Skipping {fname}, no trace found.")
        continue

    spot_word = ";".join(trace) + ";cycle{1}"

    # Debug: print first Spot word
    if total_count == 0:
        print(f"\n🔎 Sample Spot word for {fname}:\n{spot_word}\n")

    # Write Spot trace to output_dir
    spot_file = os.path.join(output_dir, fname.replace(".txt", "_spot.txt"))
    with open(spot_file, "w") as f:
        f.write(spot_word + "\n")

    # Run autfilt
    try:
        result = subprocess.run(
            ["autfilt", hoa_file, f"--accept-word={spot_word}"],
            check=False,
            text=True,
            capture_output=True,
        )
        total_count += 1
        if result.returncode == 0:
            valid_count += 1
            print(f"✅ {fname}: ACCEPTED")
        else:
            print(f"❌ {fname}: REJECTED")
    except FileNotFoundError:
        print("Error: autfilt not found in PATH.")
        sys.exit(1)

if total_count > 0:
    percent = (valid_count / total_count) * 100
    print(f"\n📊 {valid_count}/{total_count} valid traces ({percent:.2f}%)")
else:
    print("No valid trace files processed.")
