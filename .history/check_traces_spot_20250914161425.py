def Check_Traces(
    tlsf_file,
    hoa_file="controller.hoa",
    trace_dir="HOA_Traces",
    output_dir="ParsedTraces",
):
    """
    Check all traces in trace_dir against the automaton in hoa_file.
    Inputs + outputs are extracted automatically from the TLSF file via syfco.
    Spot-encoded traces are written into output_dir and overall validity % is reported.
    """
    # Extract APs (inputs then outputs)
    inputs = (
        subprocess.run(
            ["syfco", "--print-input-signals", tlsf_file],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(" ", "")
        .strip()
    )

    outputs = (
        subprocess.run(
            ["syfco", "--print-output-signals", tlsf_file],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.replace(" ", "")
        .strip()
    )

    aps = [ap for ap in (inputs + "," + outputs).split(",") if ap]

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

        # Save Spot trace
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
