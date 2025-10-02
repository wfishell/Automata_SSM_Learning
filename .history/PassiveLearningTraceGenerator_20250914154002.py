import os
import sys

def convert_trace(trace: str):
    """
    Convert a single trace string into 0/1 form.
    APs are inferred in the order they first appear in the trace.
    """
    # Remove cycle markers
    trace = trace.split("cycle{")[0].strip()

    steps = trace.split(";")
    steps = [s.strip() for s in steps if s.strip()]

    # infer APs from the first step, in order of appearance
    ap_literals = steps[0].split("&")
    aps_order = [ap.lstrip("!") for ap in ap_literals]

    converted_steps = []
    for step in steps:
        literals = step.split("&")
        vals = []
        for ap in aps_order:
            if f"!{ap}" in literals:
                vals.append("0")
            elif ap in literals:
                vals.append("1")
            else:
                vals.append("0")  # default if missing
        converted_steps.append(",".join(vals))

    # join with semicolons and ensure one final semicolon at the end
    return ";".join(converted_steps) + ";"


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_traces.py <input_dir> <output_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    with open(output_file, "w") as out_f:
        for fname in sorted(os.listdir(input_dir)):  # sorted for deterministic order
            in_path = os.path.join(input_dir, fname)
            if not os.path.isfile(in_path):
                continue

            with open(in_path) as f:
                text = f.read().strip()

            converted = convert_trace(text)
            out_f.write(converted + "\n")

            print(f"Processed {fname}")

    print(f"\nAll traces written to {output_file}")


if __name__ == "__main__":
    main()