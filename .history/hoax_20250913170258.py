# run_hoax.py
import subprocess
import sys

if len(sys.argv) < 3:
    print("Usage: python run_hoax.py <hoa_file> <config_file>")
    sys.exit(1)

hoa_file = sys.argv[1]
config_file = sys.argv[2]

try:
    result = subprocess.run(
        ["hoax", hoa_file, "--config", config_file],
        check=True,
        text=True,
        capture_output=True,
    )
except subprocess.CalledProcessError as e:
    print("❌ Error running hoax:")
    print(e.stderr)
    sys.exit(1)

with open("hoax_raw.txt", "w") as f:
    f.write(result.stdout)

print("✅ Hoax output written to hoax_raw.txt")
