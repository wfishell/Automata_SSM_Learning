import os
import sys
import re

def convert_trace(trace: str, aps_order):
    """
    Convert a single trace string to 0/1 form.
    aps_order: list of atomic propositions in the order you want them to appear.
    """
    # Remove cycle markers
    trace = trace.split("cycle{")[0].strip()

    steps = trace.split(";")
    converted_steps = []

    for step in steps:
        step = step.strip()
        if not step:
            continue

        vals = []
        for ap in aps_order:
            # Look for exact matches: 'ap' or '!ap'
            if f"!{ap}" in step.split("&"):
                vals.append("0")
            elif ap in step.split("&"):
                vals.append("1")
            else:
                # if the AP wasn't mentioned, default to 0
                vals.append("0")
        converted_steps.append(",".join(vals))

    return ";".join(converted_steps)


def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_traces.py <input_dir> <output_dir> <comma_separated_APs>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    aps_order = sys.argv[3].split(",")

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        with open(in_path) as f:
            text = f.read().strip()

        converted = convert_trace(text, aps_order)

        with open(out_path, "w") as f:
            f.write(converted + "\n")

        print(f"Converted {fname} -> {out_path}")


if __name__ == "__main__":
    main()
