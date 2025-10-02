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
