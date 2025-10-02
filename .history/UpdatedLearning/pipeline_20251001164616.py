import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib:" + os.environ.get("LD_LIBRARY_PATH", "")


def run_ltlsynt(tlsf_file: Path, hoa_file: Path):
    """Run ltlsynt on a TLSF file, strip the first line, and save HOA."""
    cmd = ["ltlsynt", "--tlsf", str(tlsf_file)]
    logging.debug(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Drop first line (header)
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

    # Drop last line
    lines = res.stdout.strip().splitlines()[:-1]
    content = "\n".join(lines)

    # Replace set() with {!ap...}
    replacement = make_replacement(aps)
    content = re.sub(r"set\(\)", replacement, content)

    hoax_file.write_text(content)


def generate_trace(hoax_file: Path, aps):
    """Generate Spot-style trace string from hoax output."""
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
    logger.info(f"[+] Automaton stats for {hoa_file}")
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


def extract_hoa_aps(hoa_content: str):
    """Extract atomic propositions (APs) from the HOA file."""
    for line in hoa_content.split("\n"):
        if line.startswith("AP:"):
            parts = line.split()
            ap_count = int(parts[1])
            return [parts[i].strip('"') for i in range(2, 2 + ap_count)]
    return []


def run_complement(hoa_file: Path, rejector_file: Path):
    """Generate rejector automaton using autfilt --complement."""
    with open(rejector_file, "w") as f:
        subprocess.run(["autfilt", "--complement", str(hoa_file)], stdout=f, check=True)


def pipeline_from_tlsf(tlsf_file: str, config_file: str):
    tlsf_path = Path(tlsf_file)
    base = tlsf_path.stem

    # Create results/<specname> directory
    results_dir = Path("results") / base
    results_dir.mkdir(parents=True, exist_ok=True)

    hoa_file = results_dir / "01-system.hoa"
    hoax_file = results_dir / "02-hoax.cleaned.hoa"
    pos_trace_file = results_dir / "03-positive-trace.spot.txt"
    sys_stats_file = results_dir / "04-system.stats.txt"
    sys_accept_file = results_dir / "05-system.accepted.hoa"

    rejector_file = results_dir / "06-rejector.hoa"
    rejector_hoax_file = results_dir / "07-rejector-hoax.cleaned.hoa"
    neg_trace_file = results_dir / "08-negative-trace.spot.txt"
    rej_stats_file = results_dir / "09-rejector.stats.txt"
    rej_accept_file = results_dir / "10-rejector.accepted.hoa"

    log_file = results_dir / "validation.log"

    try:
        # === Synthesize system automaton ===
        logger.info(f"[+] Synthesizing from TLSF {tlsf_file}")
        run_ltlsynt(tlsf_path, hoa_file)

        aps = extract_hoa_aps(hoa_file.read_text())

        # === Generate Positive trace from system ===
        logger.info("[+] Running hoax on system")
        run_hoax(hoa_file, hoax_file, Path(config_file), aps)

        pos_trace = generate_trace(hoax_file, aps)
        pos_trace_file.write_text(pos_trace)

        logger.info("[+] Checking system stats")
        run_autfilt_stats(hoa_file, sys_stats_file)

        sys_accepts_pos = run_autfilt_accept(hoa_file, pos_trace, sys_accept_file)

        # === Complement to Rejector automaton ===
        logger.info("[+] Generating rejector automaton")
        run_complement(hoa_file, rejector_file)
        aps_rejector = extract_hoa_aps(rejector_file.read_text())

        logger.info("[+] Running hoax on rejector")
        run_hoax(rejector_file, rejector_hoax_file, Path(config_file), aps_rejector)

        neg_trace = generate_trace(rejector_hoax_file, aps_rejector)
        neg_trace_file.write_text(neg_trace)

        logger.info("[+] Checking rejector stats")
        run_autfilt_stats(rejector_file, rej_stats_file)

        rej_accepts_neg = run_autfilt_accept(rejector_file, neg_trace, rej_accept_file)

        # === Cross validation ===
        sys_accepts_neg = run_autfilt_accept(hoa_file, neg_trace, Path(results_dir / "11-system.checks.neg"))
        rej_accepts_pos = run_autfilt_accept(rejector_file, pos_trace, Path(results_dir / "12-rejector.checks.pos"))

        # Write validation log
        with open(log_file, "w") as f:
            f.write("Positive trace:\n")
            f.write(f"  Accepted by system:   {sys_accepts_pos}\n")
            f.write(f"  Accepted by rejector: {rej_accepts_pos}\n\n")

            f.write("Negative trace:\n")
            f.write(f"  Accepted by system:   {sys_accepts_neg}\n")
            f.write(f"  Accepted by rejector: {rej_accepts_neg}\n")

    except Exception:
        logger.exception("Pipeline failed")
        raise

    return {
        "system": hoa_file.read_text(),
        "rejector": rejector_file.read_text(),
        "positive_trace": pos_trace,
        "sys_accepts_positive": sys_accepts_pos,
        "rej_accepts_positive": rej_accepts_pos,
        "negative_trace": neg_trace,
        "sys_accepts_negative": sys_accepts_neg,
        "rej_accepts_negative": rej_accepts_neg,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: pipeline_tlsf.py <spec.tlsf> <config.toml>")
        sys.exit(1)

    result = pipeline_from_tlsf(sys.argv[1], sys.argv[2])
    logger.info(result)
