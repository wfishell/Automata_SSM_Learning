# run_hoax_multi.py
import subprocess
import sys
MultipleTraceGenerator
import os

if len(sys.argv) < 4:
    print("Usage: python run_hoax_multi.py <hoa_file> <config_file> <num_runs>")
    sys.exit(1)

hoa_file = sys.argv[1]
config_file = sys.argv[2]

try:
    num_runs = int(sys.argv[3])
except ValueError:
    print("❌ num_runs must be an integer")
    sys.exit(1)

# Create output directory if it doesn’t exist
output_dir = "HOA_Traces"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, num_runs + 1):
    print(f"▶️ Run {i} of {num_runs}...")
    try:
        result = subprocess.run(
            ["hoax", hoa_file, "--config", config_file],
            check=True,
            text=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in run {i}:")
        print(e.stderr)
        continue

    out_file = os.path.join(output_dir, f"{i}.txt")
    with open(out_file, "w") as f:
        f.write(result.stdout)

    print(f"✅ Wrote output to {out_file}")

print("All runs completed.")
