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


def extract_hoa_aps(hoa_content: str):
    """Extract atomic propositions (APs) from the HOA file."""
    for line in hoa_content.split("\n"):
        if line.startswith("AP:"):
            parts = line.split()
            ap_count = int(parts[1])
            return [parts[i].strip('"') for i in range(2, 2 + ap_count)]
    return []


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


def trace_from_hoa(hoa_path: str, config_path: str, out_dir: str = "results"):
    """Take an HOA file, run hoax, and generate a positive trace."""
    hoa_file = Path(hoa_path)
    config_file = Path(config_path)

    results_dir = Path(out_dir) / hoa_file.stem
    results_dir.mkdir(parents=True, exist_ok=True)

    hoax_file = results_dir / "01-hoax.cleaned.txt"
    trace_file = results_dir / "02-positive-trace.spot.txt"

    aps = extract_hoa_aps(hoa_file.read_text())

    logger.info("[+] Running hoax on system automaton")
    run_hoax(hoa_file, hoax_file, config_file, aps)

    logger.info("[+] Generating trace")
    pos_trace = generate_trace(hoax_file, aps)
    trace_file.write_text(pos_trace)

    return pos_trace


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python trace_from_hoa.py <system.hoa> <config.toml>")
        sys.exit(1)

    trace = trace_from_hoa(sys.argv[1], sys.argv[2])
    logger.info(f"Generated trace:\n{trace}")
