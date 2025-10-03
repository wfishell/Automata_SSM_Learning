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


def load_dataset(trace_file: str, inputs: List[str], outputs: List[str]):
    lines = Path(trace_file).read_text().splitlines()
    dataset = []
    for line in lines:
        trace = parse_trace(line, inputs, outputs)
        prefix_closed = make_prefix_closed(trace)
        dataset.extend(prefix_closed)
    return dataset


def to_tensors(dataset, input_dim, output_dim, max_len=None):
    if max_len is None:
        max_len = max(len(x[0]) for x in dataset)

    X = torch.zeros(len(dataset), max_len, input_dim)
    Y = torch.zeros(len(dataset), max_len, output_dim)

    for i, (inp_seq, out_seq) in enumerate(dataset):
        for t, inp in enumerate(inp_seq):
            X[i, t, :] = torch.tensor(inp, dtype=torch.float32)
            Y[i, t, :] = torch.tensor(out_seq[t], dtype=torch.float32)

    return X, Y


def split_dataset(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(dataset)
    n_test = int(len(dataset) * test_ratio)
    return dataset[n_test:], dataset[:n_test]


def prepare_datasets(
    trace_file: str,
    inputs: List[str],
    outputs: List[str],
    test_ratio: float = 0.2,
    seed: int = 42,
):
    dataset = load_dataset(trace_file, inputs, outputs)
    train_set, test_set = split_dataset(dataset, test_ratio=test_ratio, seed=seed)
    X_train, Y_train = to_tensors(train_set, len(inputs), len(outputs))
    X_test, Y_test = to_tensors(test_set, len(inputs), len(outputs))
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    trace_file = "Training_Dataset.txt"
    INPUTS = ["go", "cancel", "req"]
    OUTPUTS = ["grant"]

    X_train, Y_train, X_test, Y_test = prepare_datasets(trace_file, INPUTS, OUTPUTS)

    print(X_train[0])
    print(Y_train[0])
    print("Train:", X_train.shape, Y_train.shape)
    print("Test :", X_test.shape, Y_test.shape)
