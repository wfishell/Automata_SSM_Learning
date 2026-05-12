import random
from pathlib import Path
from typing import List, Tuple

import torch


def parse_step(
    step: str, inputs: List[str], outputs: List[str]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Parse a single conjunctive step string like 'a&!p1&p0' into (input_tuple,
    output_tuple)."""
    atoms = step.split("&")
    valuation = {}
    for atom in atoms:
        if atom.startswith("!"):
            valuation[atom[1:]] = 0
        else:
            valuation[atom] = 1

    in_tuple = tuple(valuation.get(k, 0) for k in inputs)
    out_tuple = tuple(valuation.get(k, 0) for k in outputs)
    return in_tuple, out_tuple


def parse_trace(line: str, inputs: List[str], outputs: List[str]):
    line = line.strip()
    if "cycle{" in line:
        line = line.split("cycle{")[0].rstrip(";")
    steps = [s for s in line.split(";") if s]
    return [parse_step(s, inputs, outputs) for s in steps]


# ============================================================
# PREFIX-CLOSED LOADING (original behavior)
# ============================================================
def make_prefix_closed(trace):
    """
    Expand a trace into prefix-closed sequence examples:
    For each prefix of length t, we return (inputs[0:t], outputs[0:t]).
    """
    dataset = []
    for t in range(1, len(trace) + 1):
        inp_prefix = [inp for inp, _ in trace[:t]]
        out_prefix = [out for _, out in trace[:t]]
        dataset.append((tuple(inp_prefix), tuple(out_prefix)))
    return dataset


def load_dataset_prefix_closed(trace_file: str, inputs: List[str], outputs: List[str]):
    """Load traces with prefix expansion - for final-output-only supervision."""
    lines = Path(trace_file).read_text().splitlines()
    dataset = []
    for line in lines:
        trace = parse_trace(line, inputs, outputs)
        prefix_closed = make_prefix_closed(trace)
        dataset.extend(prefix_closed)
    return dataset


# ============================================================
# FULL TRACE LOADING (for Mealy seq2seq training)
# ============================================================
def load_dataset_full_traces(trace_file: str, inputs: List[str], outputs: List[str]):
    """Load traces WITHOUT prefix expansion - for Mealy seq2seq training."""
    lines = Path(trace_file).read_text().splitlines()
    dataset = []
    for line in lines:
        trace = parse_trace(line, inputs, outputs)
        inp_seq = tuple(inp for inp, _ in trace)
        out_seq = tuple(out for _, out in trace)
        dataset.append((inp_seq, out_seq))
    return dataset


# ============================================================
# SHARED UTILITIES
# ============================================================
def to_tensors(dataset, input_dim, output_dim, max_len=None):
    if max_len is None:
        max_len = max(len(x[0]) for x in dataset)

    X = torch.zeros(len(dataset), max_len, input_dim)
    Y = torch.zeros(len(dataset), max_len, output_dim)
    seq_lens = {}  # Track actual lengths

    for i, (inp_seq, out_seq) in enumerate(dataset):
        seq_lens[i] = len(inp_seq)
        for t, inp in enumerate(inp_seq):
            X[i, t, :] = torch.tensor(inp, dtype=torch.float32)
            Y[i, t, :] = torch.tensor(out_seq[t], dtype=torch.float32)

    return X, Y, seq_lens


def split_dataset(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    dataset = dataset.copy()  # Don't mutate original
    random.shuffle(dataset)
    n_test = int(len(dataset) * test_ratio)
    return dataset[n_test:], dataset[:n_test]


# ============================================================
# HIGH-LEVEL PREPARE FUNCTIONS
# ============================================================
def prepare_datasets(
    trace_file: str,
    inputs: List[str],
    outputs: List[str],
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Original prefix-closed preparation (for backward compatibility)."""
    dataset = load_dataset_prefix_closed(trace_file, inputs, outputs)
    train_set, test_set = split_dataset(dataset, test_ratio=test_ratio, seed=seed)
    X_train, Y_train, train_lens = to_tensors(train_set, len(inputs), len(outputs))
    X_test, Y_test, test_lens = to_tensors(test_set, len(inputs), len(outputs))
    return X_train, Y_train, X_test, Y_test, train_lens, test_lens


def prepare_datasets_mealy(
    trace_file: str,
    inputs: List[str],
    outputs: List[str],
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """Full trace preparation for Mealy seq2seq training."""
    dataset = load_dataset_full_traces(trace_file, inputs, outputs)
    train_set, test_set = split_dataset(dataset, test_ratio=test_ratio, seed=seed)
    X_train, Y_train, train_lens = to_tensors(train_set, len(inputs), len(outputs))
    X_test, Y_test, test_lens = to_tensors(test_set, len(inputs), len(outputs))
    return X_train, Y_train, X_test, Y_test, train_lens, test_lens


# ============================================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================================
load_dataset = load_dataset_prefix_closed


if __name__ == "__main__":
    trace_file = "Training_Dataset.txt"
    INPUTS = ["a", "b"]
    OUTPUTS = ["p0", "p1"]

    # Compare the two approaches
    print("=== PREFIX-CLOSED ===")
    X_train, Y_train, X_test, Y_test, train_lens, test_lens = prepare_datasets(
        trace_file, INPUTS, OUTPUTS
    )
    print("Train:", X_train.shape, Y_train.shape)
    print("Test :", X_test.shape, Y_test.shape)
    print("Sample train lens:", {k: train_lens[k] for k in list(train_lens.keys())[:5]})

    print("\n=== FULL TRACES (MEALY) ===")
    X_train, Y_train, X_test, Y_test, train_lens, test_lens = prepare_datasets_mealy(
        trace_file, INPUTS, OUTPUTS
    )
    print("Train:", X_train.shape, Y_train.shape)
    print("Test :", X_test.shape, Y_test.shape)
    print("Sample train lens:", {k: train_lens[k] for k in list(train_lens.keys())[:5]})
