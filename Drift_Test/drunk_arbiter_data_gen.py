import sys


def parse_step(step, inputs, outputs):
    """Parse one step, return dict of variable -> bool."""
    vals = {}
    for lit in step.split("&"):
        lit = lit.strip()
        neg = lit.startswith("!")
        var = lit[1:] if neg else lit
        vals[var] = not neg
    return vals


def emit_step(vals, inputs, outputs):
    """Emit step back to string."""
    parts = []
    for var in inputs + outputs:
        parts.append(f"{'!' if not vals.get(var, False) else ''}{var}")
    return "&".join(parts)


def transform_trace(trace_str, inputs, outputs):
    # find grant outputs (assume they start with 'g_')
    grant_vars = [v for v in outputs if v.startswith("g_")]
    n = len(grant_vars)

    # handle cycle notation
    cycle_part = ""
    if "cycle{" in trace_str:
        trace_str, cycle_part = trace_str.split("cycle{")
        cycle_part = "cycle{" + cycle_part

    steps = [s.strip() for s in trace_str.strip().split(";") if s.strip()]

    grants = {g: 0 for g in grant_vars}
    total = 0
    output = []

    for step in steps:
        vals = parse_step(step, inputs, outputs)
        mean = total / n if total > 0 else 0

        # who is blocked?
        blocked = {g: grants[g] >= mean + 2 for g in grant_vars}

        # which grant was given?
        granted = [g for g in grant_vars if vals.get(g, False)]

        if granted:
            orig = granted[0]
            if blocked[orig]:
                # find corresponding request variable
                # assume g_i corresponds to s_i or r_i
                def get_request(g):
                    idx = g.split("_")[1]
                    for prefix in ["s_", "r_"]:
                        if prefix + idx in inputs:
                            return prefix + idx
                    return None

                # pick eligible: requesting and not blocked
                eligible = [
                    g
                    for g in grant_vars
                    if not blocked[g] and vals.get(get_request(g), False)
                ]
                new_grant = eligible[0] if eligible else None
            else:
                new_grant = orig
        else:
            new_grant = None

        # update grant outputs
        for g in grant_vars:
            vals[g] = g == new_grant

        if new_grant:
            grants[new_grant] += 1
            total += 1

        output.append(emit_step(vals, inputs, outputs))

    return ";".join(output) + (";" + cycle_part if cycle_part else "")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python transform.py <inputs> <outputs> < traces.txt")
        print("Example: python transform.py 's_0,s_1,s_2' 'g_0,g_1,g_2'")
        sys.exit(1)

    inputs = [v.strip() for v in sys.argv[1].split(",")]
    outputs = [v.strip() for v in sys.argv[2].split(",")]

    for line in sys.stdin:
        line = line.strip()
        if line:
            print(transform_trace(line, inputs, outputs))
