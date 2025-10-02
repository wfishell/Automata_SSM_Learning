"""
Automata pipeline
Authors: Nikolaus Holzer, Will Fishell
Date: September 2025

Pipeline to convert a tlsf input file into hoa and run clis to extract reasoning traces

NOTES:
If we give it a lot of traces, it can figure out the legal transitions
"""

import json

# import getopt
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import spot

# # local imports to abstract away the corp call
from . import cause

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level, can be changed externally

os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)


def run_ltlsynt(tlsf_file: Path, hoa_file: Path):
    """Run ltlsynt on a TLSF file, strip the first line, and save HOA."""
    cmd = ["ltlsynt", "--tlsf", str(tlsf_file)]
    logging.debug(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Drop first line
    lines = res.stdout.splitlines()[1:]
    hoa_file.write_text("\n".join(lines))


def make_replacement(aps):
    """Build replacement string for empty set() in hoax output."""
    parts = [f"'!{ap}'" for ap in aps]
    return "{" + ", ".join(parts) + "}"


def run_hoax(hoa_file: Path, hoax_file: Path, config_file: Path, aps):
    """Run hoax with config, clean its output, and save result."""
    cmd = ["hoax", str(hoa_file), "--config", str(config_file)]

    res = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Drop last line (like `sed '$d'`)
    lines = res.stdout.strip().splitlines()[:-1]
    content = "\n".join(lines)

    # Replace set() with {!ap...}
    replacement = make_replacement(aps)
    content = re.sub(r"set\(\)", replacement, content)

    hoax_file.write_text(content)


def generate_trace(hoax_file: Path, aps):
    """Generate trace string from hoax output (Spot word)."""
    trace = []
    for line in hoax_file.read_text().splitlines():
        raw = re.search(r"{(.*)}", line)
        if not raw:
            continue
        present = [tok.strip("'\" ") for tok in raw.group(1).split(",") if tok.strip()]
        assignment = [ap if ap in present else f"!{ap}" for ap in aps]
        trace.append("&".join(assignment))
    return ";".join(trace) + ";cycle{1}"


def run_autfilt_stats(hoa_file: Path, stats_file: Path):
    """Show automaton stats via autfilt."""
    logger.info("[+] Automaton stats:")
    with open(stats_file, "w") as f:
        subprocess.run(
            ["autfilt", "--stats=%s states, %e edges, %a acc-sets, %c SCCs, det=%d"],
            input=hoa_file.read_text(),
            text=True,
            stdout=f,
            check=True,
        )


def run_autfilt_accept(hoa_file: Path, trace: str, output_file: Path):
    """Check acceptance of a trace in automaton using autfilt."""
    cmd = ["autfilt", f"--accept-word={trace}"]
    res = subprocess.run(cmd, input=hoa_file.read_bytes(), capture_output=True)
    output_file.write_bytes(res.stdout)
    return res.returncode == 0


def extract_outputs(tlsf_file: Path, outputs_file: Path):
    # run syfco to get outputs
    cmd = ["syfco", tlsf_file, "-outs"]
    res = subprocess.run(cmd, capture_output=True)

    output_params = res.stdout.decode("utf-8").strip().split(",")
    outputs_file.write_text(res.stdout.decode("utf-8"))

    return output_params


def extract_effects(
    trace: str,
    output_params: list,
    effects_file: Path,
):
    """Extracts the output parameters of the tlsf file.

    Returns a list of effects with the correct timestep offset
    """

    # NOTE there may be some better way of doing this, but it should be O(N)
    indexed_trace = [s.split("&") for s in trace.strip().split(";")[:-1]]

    effects_str = ""
    effects_arr = []
    for i, params in enumerate(indexed_trace):
        for output in output_params:  # is O(N) output params fixed
            if output in params:
                effects_str += f"{'X'*i} {output}\n"
                effects_arr.append(f"{'X'*i} {output}")

    effects_file.write_text(effects_str)

    return effects_arr


def extract_hoa_aps(hoa_content: str):
    """Extract the atomic propositions (APs) from the HOA file.

    Returns a list of AP names in order.
    """
    for line in hoa_content.split("\n"):
        if line.startswith("AP:"):
            # Parse line like: AP: 2 "r_0" "r_1"
            parts = line.split()
            ap_count = int(parts[1])
            aps = []
            for i in range(2, 2 + ap_count):
                ap_name = parts[i].strip('"')
                aps.append(ap_name)
            return aps
    return []


def filter_props_for_hoa(all_props: list, hoa_aps: list):
    """Filter the propositions from the trace to only include those that are relevant to
    this specific HOA automaton."""
    relevant_props = []
    for prop in all_props:
        # Remove negation to get base prop name
        base_prop = prop.lstrip("!")
        if base_prop in hoa_aps:
            relevant_props.append(prop)
    return relevant_props


def parse_hoa_states(hoa_content: str):
    """Parse HOA content and return a dictionary of states and their transitions.

    Returns: {state_id: [(condition, target_state), ...]}
    """
    states = {}
    current_state = None

    lines = hoa_content.split("\n")
    in_body = False

    for line in lines:
        line = line.strip()

        if line == "--BODY--":
            in_body = True
            continue
        elif line == "--END--":
            break

        if not in_body:
            continue

        if line.startswith("State:"):
            # Parse state line like "State: 0" or "State: 17 {0}"
            parts = line.split()
            current_state = int(parts[1])
            states[current_state] = []
        elif line.startswith("[") and current_state is not None:
            # Parse transition line like "[!0&1] 1"
            condition_end = line.find("]")
            condition = line[1:condition_end]
            target = int(line[condition_end + 1 :].strip())
            states[current_state].append((condition, target))

    return states


def evaluate_condition(condition: str, current_props: list, hoa_aps: list):
    """Evaluate if a condition matches the current propositions.

    condition: string like "!0&1" or "t" (using HOA indices)
    current_props: list of current atomic propositions (filtered for this HOA)
    hoa_aps: list of AP names in the order they appear in the HOA
    """
    if condition == "t":
        return True

    # Build a mapping of which APs are true/false in current step
    ap_values = {}
    for i, ap_name in enumerate(hoa_aps):
        # Check if this AP is present (true) or negated (false) in current_props
        if ap_name in current_props:
            ap_values[str(i)] = True
        elif f"!{ap_name}" in current_props:
            ap_values[str(i)] = False
        else:
            # If neither present nor explicitly negated, assume false
            ap_values[str(i)] = False

    # Replace indices in condition with actual boolean values
    eval_condition = condition
    for i, value in ap_values.items():
        eval_condition = eval_condition.replace(f"!{i}", f"not {value}")
        eval_condition = eval_condition.replace(i, str(value))

    # Replace logical operators
    eval_condition = eval_condition.replace("&", " and ").replace("|", " or ")

    return eval(eval_condition)


def parse_required_inputs(condition: str, hoa_aps: list):
    """Parse a condition string and return human-readable required inputs using the
    actual AP names from the HOA."""
    if condition == "t":
        return ["no constraints"]

    # Convert numeric indices back to AP names
    # Process negated indices first to avoid double replacement
    readable_condition = condition

    # Sort by index in descending order to avoid partial replacements
    for i in sorted(range(len(hoa_aps)), reverse=True):
        ap_name = hoa_aps[i]
        # Replace negated indices first
        readable_condition = readable_condition.replace(f"!{i}", f"!{ap_name}")
        # Then replace positive indices, but avoid replacing parts of AP names
        readable_condition = re.sub(rf"\b{i}\b", ap_name, readable_condition)

    # Make it more readable
    readable_condition = readable_condition.replace("&", " AND ").replace("|", " OR ")

    return [readable_condition]


def trace_through_hoa(hoa_file_path: str, trace_str: str, effect_str: str):
    """Trace through the HOA automaton using the given trace and record required input
    conditions at each step until the effect occurs.

    Returns a dictionary: {time_step: [required_conditions]}
    """
    required_inputs = {}

    # Parse the trace to get timesteps
    trace_clean = trace_str.split("cycle{")[0].rstrip(";")
    timesteps = [ts.split("&") for ts in trace_clean.split(";")]

    # Determine when the effect occurs
    effect_time = None
    x_count = effect_str.count("X")  # Number of X's indicates the timestep
    if x_count == 0:
        effect_time = 0
    else:
        effect_time = x_count

    # Read and parse the HOA file
    with open(hoa_file_path, "r") as f:
        hoa_content = f.read()

    # Extract the APs that this specific HOA uses
    hoa_aps = extract_hoa_aps(hoa_content)

    # Parse HOA content to extract states and transitions
    states = parse_hoa_states(hoa_content)

    # Find the start state
    start_state = None
    for line in hoa_content.split("\n"):
        if line.startswith("Start:"):
            start_state = int(line.split()[1])
            break

    if start_state is None:
        return required_inputs

    current_state = start_state

    # Trace through the automaton up to the effect time
    for time_step in range(min(effect_time + 1, len(timesteps))):
        current_props = timesteps[time_step]

        # Filter current props to only include those relevant to this HOA
        relevant_props = filter_props_for_hoa(current_props, hoa_aps)

        # Find which transition is taken from current state
        if current_state in states:
            transitions = states[current_state]
            next_state = None
            required_condition = None

            # Check each transition to see which one matches current props
            for condition, target_state in transitions:
                if evaluate_condition(condition, relevant_props, hoa_aps):
                    next_state = target_state
                    required_condition = condition
                    break

            if (
                required_condition and required_condition != "t"
            ):  # "t" means always true
                required_inputs[time_step] = parse_required_inputs(
                    required_condition, hoa_aps
                )
            elif required_condition == "t":
                required_inputs[time_step] = ["no constraints"]

            current_state = next_state

            # Stop if we've reached the effect time
            if time_step >= effect_time:
                break

    return required_inputs


def check_causality(
    effects_arr: list, trace_str: str, hoa_file: Path, output_file: Path, log_file: Path
):
    """Modified to save HOA files for each effect and trace through them to find
    required inputs at each timestep."""
    system = spot.automaton(str(hoa_file))
    trace = spot.parse_word(trace_str.rstrip())
    log_str = ""

    # Dictionary to store effect -> temp HOA file mapping
    effect_hoa_files = {}
    # Dictionary to store effect -> required inputs mapping
    effect_inputs = {}

    try:
        # First pass: generate HOA files for each effect
        for effect_str in effects_arr:
            effect = spot.postprocess(
                spot.translate(spot.formula("!(" + effect_str.strip() + ")")),
                "buchi",
                "state-based",
                "small",
                "high",
            )

            result = cause.synthesize(system, trace, effect, False, False)

            if result.is_empty():
                log_str += f"No cause found for {effect_str}\n"
            else:
                log_str += f"Cause found for {effect_str}\n"

                # Create temporary file for this effect's HOA
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix=".hoa",
                    prefix=f'effect_{effect_str.replace(" ", "_").replace("X", "")}_',
                )
                os.close(temp_fd)  # Close the file descriptor

                # Write HOA content to temp file
                with open(temp_path, "w") as temp_file:
                    temp_file.write(result.to_str())

                effect_hoa_files[effect_str] = temp_path
                logger.info(f"Saved HOA for {effect_str} to {temp_path}")

        # Second pass: trace through each effect's HOA to find required inputs
        for effect_str, hoa_path in effect_hoa_files.items():
            required_inputs = trace_through_hoa(hoa_path, trace_str, effect_str)
            effect_inputs[effect_str] = required_inputs

        # Write log
        with open(log_file, "w") as f:
            f.write(log_str)
            f.write("\n--- Required Inputs Analysis ---\n")
            for effect_str, inputs in effect_inputs.items():
                f.write(f"\nFor effect {effect_str}:\n")
                for time_step, conditions in inputs.items():
                    f.write(f"  Time {time_step}: {conditions}\n")

        with open(output_file, "w") as f:
            json.dump(effect_inputs, f, indent=4)

        return effect_inputs

    finally:
        # Clean up temporary files
        for temp_path in effect_hoa_files.values():
            try:
                os.unlink(temp_path)
            except OSError as e:
                logging.warning(f"Could not delete temp file {temp_path}: {e}")


def pipeline(tlsf_file: str, config_file: str):
    tlsf_path = Path(tlsf_file)
    base = tlsf_path.stem

    # Create results/<specname> directory
    results_dir = Path("results") / base
    results_dir.mkdir(parents=True, exist_ok=True)

    hoa_file = results_dir / "01-system.hoa"
    hoax_file = results_dir / "02-hoax.cleaned.hoa"
    trace_file = results_dir / "03-trace.spot.txt"
    stats_file = results_dir / "04-autfilt.stats.txt"
    accepted_file = results_dir / "05-autfilt.accepted.hoa"
    effects_file = results_dir / "06-effects.txt"
    outputs_file = results_dir / "07-outputs.txt"
    causal_file = results_dir / "08-causal.json"
    acceptance_log_file = results_dir / "acceptance.log"
    corp_log_file = results_dir / "corp.log"

    try:
        logger.info(f"[+] Running ltlsynt on {tlsf_file}")
        run_ltlsynt(tlsf_path, hoa_file)

        aps = extract_hoa_aps(hoa_file.read_text())

        logger.info(f"[+] Running hoax on {hoa_file}")
        run_hoax(hoa_file, hoax_file, Path(config_file), aps)

        logger.info("[+] Generating trace")
        trace = generate_trace(hoax_file, aps)
        trace_file.write_text(trace)

        logger.info("[+] Automaton stats:")
        run_autfilt_stats(hoa_file, stats_file)

        logger.info("[+] Checking acceptance")
        accepted = run_autfilt_accept(hoa_file, trace, accepted_file)

        with open(acceptance_log_file, "w") as f:
            f.write("Pass.\n" if accepted else "Did not pass.\n")

    
    return {
        "hoa": hoa_file.read_text(),
        "aps": aps,
        "trace": trace,
        "accepted": accepted,
    }


# This should just run from main.py
if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: pipeline.py <spec.tlsf>")
        sys.exit(1)

    result = pipeline(
        sys.argv[1], os.path.join(os.path.dirname(__file__), "random_config.py")
    )
    logger.info(result)