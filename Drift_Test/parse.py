#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def clean_training_output(data):
    """Extract epoch data from training_output strings."""
    for result in data:
        if result.get("training_output"):
            epochs = []
            for match in re.finditer(
                r"Epoch\s+(\d+)\s+\|\s+Loss\s+=\s+([\d.]+)\s+\|\s+Train Step\s+=\s+([\d.]+)\s+\|\s+Train Trace\s+=\s+([\d.]+)\s+\|\s+Test Step\s+=\s+([\d.]+)\s+\|\s+Test Trace\s+=\s+([\d.]+)",
                result["training_output"],
            ):
                epochs.append(
                    {
                        "epoch": int(match.group(1)),
                        "loss": float(match.group(2)),
                        "train_step": float(match.group(3)),
                        "train_trace": float(match.group(4)),
                        "test_step": float(match.group(5)),
                        "test_trace": float(match.group(6)),
                    }
                )
            result["epochs"] = epochs
            del result["training_output"]
    return data


def main():
    parser = argparse.ArgumentParser(description="Clean training results JSON files")
    parser.add_argument("files", nargs="+", help="JSON files to clean")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    args = parser.parse_args()

    for filepath in args.files:
        path = Path(filepath)

        with open(path, "r") as f:
            data = json.load(f)

        cleaned = clean_training_output(data)

        output_dir = Path(args.output_dir) if args.output_dir else path.parent
        output_path = output_dir / f"clean_{path.name}"

        with open(output_path, "w") as f:
            json.dump(cleaned, f, indent=2)

        print(f"Cleaned: {path} -> {output_path}")


if __name__ == "__main__":
    main()
