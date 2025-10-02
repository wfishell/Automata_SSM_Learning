import re
import sys

def parse_dot(dot_file):
    """Parse DOT file with Mealy-style transitions (input / output)."""
    states = set()
    transitions = {}

    with open(dot_file) as f:
        for line in f:
            line = line.strip()
            # Transition line like: 0 -> 1 [label="cond / output"]
            match = re.match(r'(\d+)\s*->\s*(\d+)\s*\[label="(.+)"\]', line)
            if match:
                src, dst, label = match.groups()
                src, dst = int(src), int(dst)

                if src not in transitions:
                    transitions[src] = []
                transitions[src].append((label, dst))

                states.add(src)
                states.add(dst)

    return sorted(states), transitions


def write_hoa(states, transitions, inputs, outputs, out_file):
    """Write HOA automaton with Mealy-like input/output labels."""
    with open(out_file, "w") as f:
        f.write("HOA: v1\n")
        f.write(f"States: {len(states)}\n")
        f.write("Start: 0\n")
        f.write("AP: {} {}\n".format(len(inputs) + len(outputs),
            " ".join([f"\"{x}\"" for x in inputs + outputs])))
        f.write("acc-name: all\n")
        f.write("Acceptance: 0 t\n")
        f.write("properties: trans-labels explicit-labels deterministic\n")
        f.write("--BODY--\n")

        for s in states:
            f.write(f"State: {s}\n")
            for label, dst in transitions.get(s, []):
                # Split "input / output"
                if "/" in label:
                    inpart, outpart = map(str.strip, label.split("/"))
                    cond = f"{inpart} & {outpart}" if inpart else outpart
                else:
                    cond = label.strip()
                f.write(f"[{cond}] {dst}\n")

        f.write("--END--\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dot2hoa.py input.dot output.hoa")
        sys.exit(1)

    dot_file, out_file = sys.argv[1], sys.argv[2]

    # Adjust these to match your signals
    inputs = ["r_0", "r_1", "r_2"]
    outputs = ["g_0", "g_1", "g_2"]

    states, transitions = parse_dot(dot_file)
    write_hoa(states, transitions, inputs, outputs, out_file)

    print(f"Converted {dot_file} → {out_file}")
