#!/usr/bin/env python3
"""Run drifting_arbiter.py and Trace_Checker.py for varying N values, collect accuracy
results, and plot."""

import re
import subprocess

import matplotlib.pyplot as plt


def run_experiment(n_value, num_traces=1000):
    """Run both scripts for a given N value and return accuracy."""

    # Run drifting_arbiter.py
    cmd1 = [
        "python",
        "drifting_arbiter.py",
        str(num_traces),
        "traces.txt",
        str(n_value),
    ]
    subprocess.run(cmd1, check=True)

    # Run Trace_Checker.py and capture output
    cmd2 = ["python", "Trace_Checker.py", "System.hoa", "traces.txt"]
    result = subprocess.run(cmd2, capture_output=True, text=True, check=True)

    # Parse accuracy from output (adjust regex based on actual output format)
    output = result.stdout
    match = re.search(r"[Aa]ccuracy[:\s]+(\d+\.?\d*)", output)
    if match:
        return float(match.group(1))

    # Try parsing as percentage
    match = re.search(r"(\d+\.?\d*)\s*%", output)
    if match:
        return float(match.group(1))

    # Try parsing fraction like "45/50"
    match = re.search(r"(\d+)\s*/\s*(\d+)", output)
    if match:
        return float(match.group(1)) / float(match.group(2)) * 100

    print(f"Warning: Could not parse accuracy from output:\n{output}")
    return None


def main():
    n_values = list(range(5, 55, 5))  # 5, 10, 15, ..., 50
    accuracies = []

    print("Running experiments...")
    for n in n_values:
        print(f"  N = {n}...", end=" ", flush=True)
        acc = run_experiment(n)
        accuracies.append(acc)
        print(f"Accuracy: {acc}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, accuracies, marker="o", linewidth=2, markersize=8)
    plt.xlabel("N", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy vs N", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(n_values)
    plt.tight_layout()
    plt.savefig("accuracy_vs_n.png", dpi=150)
    plt.show()

    # Save results to file
    with open("results.txt", "w") as f:
        f.write("N\tAccuracy\n")
        for n, acc in zip(n_values, accuracies):
            f.write(f"{n}\t{acc}\n")

    print("\nResults saved to results.txt")
    print("Plot saved to accuracy_vs_n.png")


if __name__ == "__main__":
    main()
