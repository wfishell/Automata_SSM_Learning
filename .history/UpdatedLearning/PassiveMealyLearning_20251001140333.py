import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_i_o_traces.py <trace_file>")
        sys.exit(1)

    trace_file = sys.argv[1]  # first argument after the script name

    INPUTS = ["r_0", "r_1"]
    OUTPUTS = ["g_0", "g_1"]

    dataset = process_file(trace_file, INPUTS, OUTPUTS)

    learned_mealy = run_RPNI(dataset, automaton_type="mealy")

    print("\n[+] Learned Mealy Machine:")
    print(learned_mealy)

    # use same folder as input file for output
    out_file = str(Path(trace_file).with_suffix(".hoa"))
    save_mealy_as_hoa(learned_mealy, INPUTS, OUTPUTS, out_file)
