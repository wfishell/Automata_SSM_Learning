import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_hoa_stats(hoa_content: str) -> dict:
    """Parse HOA format to extract stats."""
    stats = {
        "num_states": None,
        "num_inputs": None,
        "num_outputs": None,
        "input_aps": [],
        "output_aps": [],
    }

    # Extract number of states: "States: N"
    states_match = re.search(r"^States:\s*(\d+)", hoa_content, re.MULTILINE)
    if states_match:
        stats["num_states"] = int(states_match.group(1))

    # Extract APs: "AP: N "ap1" "ap2" ..."
    ap_match = re.search(r"^AP:\s*(\d+)\s*(.*)", hoa_content, re.MULTILINE)
    if ap_match:
        ap_str = ap_match.group(2)
        aps = re.findall(r'"([^"]+)"', ap_str)
        stats["all_aps"] = aps

    # Extract controllable-AP (outputs): "controllable-AP: 0 1 2..."
    ctrl_match = re.search(r"^controllable-AP:\s*([\d\s]*)", hoa_content, re.MULTILINE)
    if ctrl_match and ap_match:
        ctrl_indices = [int(x) for x in ctrl_match.group(1).split() if x.strip()]
        aps = stats.get("all_aps", [])
        stats["output_aps"] = [aps[i] for i in ctrl_indices if i < len(aps)]
        stats["input_aps"] = [ap for i, ap in enumerate(aps) if i not in ctrl_indices]
        stats["num_inputs"] = len(stats["input_aps"])
        stats["num_outputs"] = len(stats["output_aps"])

    return stats


def run_ltlsynt(tlsf_path: Path, timeout: int = 60) -> dict:
    """Run ltlsynt on a TLSF file and return stats."""
    result = {
        "tlsf_file": str(tlsf_path),
        "tlsf_name": tlsf_path.stem,
        "success": False,
        "error": None,
        "num_states": None,
        "num_inputs": None,
        "num_outputs": None,
        "input_aps": [],
        "output_aps": [],
        "realizable": None,
    }

    try:
        # Run ltlsynt
        proc = subprocess.run(
            ["ltlsynt", "--tlsf", str(tlsf_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = proc.stdout
        stderr = proc.stderr

        # Check realizability
        if "REALIZABLE" in stdout:
            result["realizable"] = True
        elif "UNREALIZABLE" in stdout:
            result["realizable"] = False
            result["success"] = True  # Successfully determined unrealizable
            return result

        # Parse HOA output
        if "HOA:" in stdout:
            stats = parse_hoa_stats(stdout)
            result.update(stats)
            result["success"] = True
        else:
            result["error"] = f"No HOA output. stderr: {stderr[:200]}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
    except FileNotFoundError:
        result["error"] = "ltlsynt not found - install spot"
    except Exception as e:
        result["error"] = str(e)

    return result


def process_benchmarks(
    benchmark_dir: Path, output_file: Path, timeout: int = 60, max_workers: int = 4
):
    """Process all TLSF files in directory."""

    tlsf_files = list(benchmark_dir.glob("**/*.tlsf"))
    print(f"Found {len(tlsf_files)} TLSF files")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_ltlsynt, tlsf_path, timeout): tlsf_path
            for tlsf_path in tlsf_files
        }

        for i, future in enumerate(as_completed(futures)):
            tlsf_path = futures[future]
            try:
                result = future.result()
                results.append(result)

                status = "✓" if result["success"] else "✗"
                states = result.get("num_states", "?")
                inputs = result.get("num_inputs", "?")
                outputs = result.get("num_outputs", "?")
                print(
                    f"[{i+1}/{len(tlsf_files)}] {status} {tlsf_path.name}: "
                    f"states={states}, inputs={inputs}, outputs={outputs}"
                )

            except Exception as e:
                print(f"[{i+1}/{len(tlsf_files)}] ✗ {tlsf_path.name}: {e}")
                results.append(
                    {
                        "tlsf_file": str(tlsf_path),
                        "tlsf_name": tlsf_path.stem,
                        "success": False,
                        "error": str(e),
                    }
                )

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")

    # Also save CSV summary
    csv_file = output_file.with_suffix(".csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tlsf_name",
                "success",
                "realizable",
                "num_states",
                "num_inputs",
                "num_outputs",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "tlsf_name": r.get("tlsf_name"),
                    "success": r.get("success"),
                    "realizable": r.get("realizable"),
                    "num_states": r.get("num_states"),
                    "num_inputs": r.get("num_inputs"),
                    "num_outputs": r.get("num_outputs"),
                    "error": r.get("error"),
                }
            )
    print(f"Saved CSV to {csv_file}")

    # Print summary
    successful = [r for r in results if r["success"]]
    realizable = [r for r in results if r.get("realizable")]
    unrealizable = [r for r in results if r.get("realizable")]

    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Total files:    {len(results)}")
    print(f"  Successful:     {len(successful)}")
    print(f"  Realizable:     {len(realizable)}")
    print(f"  Unrealizable:   {len(unrealizable)}")
    print(f"  Failed/Timeout: {len(results) - len(successful)}")

    if realizable:
        states = [r["num_states"] for r in realizable if r.get("num_states")]
        inputs = [r["num_inputs"] for r in realizable if r.get("num_inputs")]
        outputs = [r["num_outputs"] for r in realizable if r.get("num_outputs")]

        if states:
            print("\nRealizable benchmarks stats:")
            print(
                f"  States:  min={min(states)}, max={max(states)}, avg={sum(states)/len(states):.1f}"
            )
        if inputs:
            print(
                f"  Inputs:  min={min(inputs)}, max={max(inputs)}, avg={sum(inputs)/len(inputs):.1f}"
            )
        if outputs:
            print(
                f"  Outputs: min={min(outputs)}, max={max(outputs)}, avg={sum(outputs)/len(outputs):.1f}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ltlsynt on TLSF benchmarks")
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        nargs="?",
        default=Path("/workspaces/Automata_SSM_Learning/TestSet/benchmarks"),
        help="Directory containing TLSF files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tlsf_synth_results.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=60,
        help="Timeout per file in seconds",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs",
    )

    args = parser.parse_args()

    if not args.benchmark_dir.exists():
        print(f"Error: Directory not found: {args.benchmark_dir}")
        exit(1)

    process_benchmarks(args.benchmark_dir, args.output, args.timeout, args.jobs)
