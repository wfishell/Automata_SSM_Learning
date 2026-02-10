import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class HOAParser:
    """Parse HOA files without spot."""

    def __init__(self, hoa_file: str):
        self.hoa_file = hoa_file
        self.num_states = 0
        self.initial_state = 0
        self.ap_list = []  # Ordered list of atomic propositions
        self.transitions = {}  # state -> list of (condition_str, dest_state)
        self.acc_sets = {}  # state -> acceptance set (if any)

        self._parse()

    def _parse(self):
        """Parse the HOA file."""
        with open(self.hoa_file, "r") as f:
            content = f.read()

        lines = content.strip().split("\n")
        in_body = False
        current_state = None

        for line in lines:
            line = line.strip()

            if not line or line.startswith("HOA:") or line.startswith("tool:"):
                continue

            if line.startswith("States:"):
                self.num_states = int(line.split(":")[1].strip())

            elif line.startswith("Start:"):
                self.initial_state = int(line.split(":")[1].strip())

            elif line.startswith("AP:"):
                # Parse: AP: 3 "a" "b" "c"
                match = re.search(r"AP:\s*\d+\s+(.*)", line)
                if match:
                    ap_str = match.group(1)
                    self.ap_list = re.findall(r'"([^"]*)"', ap_str)

            elif line == "--BODY--":
                in_body = True

            elif line == "--END--":
                in_body = False

            elif in_body:
                if line.startswith("State:"):
                    # Parse: State: 0 or State: 0 {0}
                    parts = line.split()
                    current_state = int(parts[1])
                    self.transitions[current_state] = []
                    # Check for acceptance set
                    if "{" in line:
                        acc_match = re.search(r"\{(\d+)\}", line)
                        if acc_match:
                            self.acc_sets[current_state] = int(acc_match.group(1))

                elif current_state is not None and line.startswith("["):
                    # Parse transition: [condition] dest or [condition] dest {acc}
                    match = re.match(r"\[([^\]]*)\]\s*(\d+)", line)
                    if match:
                        condition = match.group(1)
                        dest = int(match.group(2))
                        self.transitions[current_state].append((condition, dest))

    def evaluate_condition(self, condition: str, valuation: Dict[str, int]) -> bool:
        """Evaluate a HOA condition string against a valuation.

        Condition format: "0&!1" means AP[0]=true AND AP[1]=false
        Also handles: "t" (true), "f" (false), "|" (or)
        """
        if condition == "t":
            return True
        if condition == "f":
            return False

        # Handle OR (|) by splitting and checking if any clause is true
        if "|" in condition:
            clauses = condition.split("|")
            return any(self.evaluate_condition(c.strip(), valuation) for c in clauses)

        # Handle AND (&) - all conjuncts must be true
        if "&" in condition:
            conjuncts = condition.split("&")
        else:
            conjuncts = [condition]

        for conj in conjuncts:
            conj = conj.strip()
            if not conj:
                continue

            negated = conj.startswith("!")
            if negated:
                conj = conj[1:]

            # conj should now be an integer index into ap_list
            try:
                ap_index = int(conj)
                ap_name = self.ap_list[ap_index]
                ap_value = valuation.get(ap_name, 0)

                if negated:
                    if ap_value != 0:
                        return False
                else:
                    if ap_value == 0:
                        return False
            except (ValueError, IndexError):
                # If we can't parse it, skip
                continue

        return True


class MealyMachine:
    """Mealy machine built from parsed HOA file."""

    def __init__(self, hoa_file: str, input_aps: List[str], output_aps: List[str]):
        self.input_aps = input_aps
        self.output_aps = output_aps

        # Parse HOA
        self.hoa = HOAParser(hoa_file)
        self.n_states = self.hoa.num_states
        self.initial_state = self.hoa.initial_state
        self.ap_list = self.hoa.ap_list

        # Build explicit transition table
        # (state, input_tuple) -> (next_state, output_tuple)
        self.transitions = {}
        self._build_transition_table()

    def _build_transition_table(self):
        """Build explicit transition table by enumerating all input combinations."""
        n_inputs = len(self.input_aps)

        # Generate all possible input combinations
        all_inputs = []
        for i in range(2**n_inputs):
            inp = tuple((i >> j) & 1 for j in range(n_inputs))
            all_inputs.append(inp)

        for state in range(self.n_states):
            for inp in all_inputs:
                # Build valuation for inputs
                valuation = {ap: inp[i] for i, ap in enumerate(self.input_aps)}

                # Find matching transition
                for condition, dest in self.hoa.transitions.get(state, []):
                    # We need to find a complete valuation (inputs + outputs) that satisfies condition
                    # Try all output combinations
                    found = False
                    for out_i in range(2 ** len(self.output_aps)):
                        out = tuple(
                            (out_i >> j) & 1 for j in range(len(self.output_aps))
                        )

                        # Complete valuation
                        full_val = dict(valuation)
                        for j, ap in enumerate(self.output_aps):
                            full_val[ap] = out[j]

                        if self.hoa.evaluate_condition(condition, full_val):
                            self.transitions[(state, inp)] = (dest, out)
                            found = True
                            break

                    if found:
                        break

    def step(
        self, state: int, input_tuple: Tuple[int, ...]
    ) -> Tuple[int, Tuple[int, ...]]:
        """Execute one step: returns (next_state, output)."""
        key = (state, input_tuple)
        if key in self.transitions:
            return self.transitions[key]
        else:
            raise ValueError(f"No transition for state {state}, input {input_tuple}")

    def run_sequence(
        self, input_sequence: List[Tuple[int, ...]]
    ) -> Tuple[int, List[Tuple[int, ...]]]:
        """Run a sequence of inputs.

        Returns: (final_state, list_of_outputs)
        """
        state = self.initial_state
        outputs = []

        for inp in input_sequence:
            state, out = self.step(state, inp)
            outputs.append(out)

        return state, outputs

    def get_state_after_sequence(self, input_sequence: List[Tuple[int, ...]]) -> int:
        """Get the state reached after processing the input sequence."""
        if len(input_sequence) == 0:
            return self.initial_state
        final_state, _ = self.run_sequence(input_sequence)
        return final_state

    def print_transition_table(self):
        """Print the transition table for debugging."""
        print(f"\nMealy Machine: {self.n_states} states")
        print(f"Initial state: {self.initial_state}")
        print(f"Input APs: {self.input_aps}")
        print(f"Output APs: {self.output_aps}")
        print(f"AP order in HOA: {self.ap_list}")
        print("\nTransition table:")

        for (state, inp), (next_state, out) in sorted(self.transitions.items()):
            inp_str = "".join(str(b) for b in inp)
            out_str = "".join(str(b) for b in out)
            print(f"  q{state} --[in={inp_str}]--> q{next_state} / out={out_str}")


class EquivalenceAnalyzer:
    """Analyzes whether SSM hidden states align with Myhill-Nerode equivalence classes.

    The equivalence class of an input sequence = the Mealy state it reaches.
    """

    def __init__(self, ssm_model, mealy: MealyMachine, device="cpu"):
        self.model = ssm_model
        self.mealy = mealy
        self.device = device
        self.input_aps = mealy.input_aps

    def generate_sequences(
        self, num_samples: int, max_len: int
    ) -> List[List[Tuple[int, ...]]]:
        """Generate random input sequences."""
        n_inputs = len(self.input_aps)
        sequences = []

        for _ in range(num_samples):
            length = np.random.randint(1, max_len + 1)
            seq = [
                tuple(np.random.randint(0, 2, n_inputs).tolist()) for _ in range(length)
            ]
            sequences.append(seq)

        return sequences

    def get_ssm_hidden_state(self, input_sequence: List[Tuple[int, ...]]) -> np.ndarray:
        """Run SSM on input sequence, return final hidden state."""
        self.model.eval()

        with torch.no_grad():
            # Convert to tensor: (1, T, input_dim)
            seq_array = np.array(input_sequence, dtype=np.float32)
            x = torch.tensor(seq_array).unsqueeze(0).to(self.device)

            B, T, _ = x.shape
            x_embed = torch.relu(self.model.embed(x))
            h = self.model.h0.unsqueeze(0).expand(B, -1)

            for t in range(T):
                h = torch.tanh(h @ self.model.A.T + x_embed[:, t] @ self.model.B.T)

            return h.squeeze(0).cpu().numpy()

    def analyze(self, num_samples: int = 500, max_len: int = 10) -> Dict:
        """
        Main analysis: compare SSM hidden states to equivalence classes.

        Returns dict with:
        - adjusted_rand_index: How well clusters match equivalence classes
        - normalized_mutual_info: Information-theoretic alignment
        - separation_ratio: Between-class / within-class variance
        - hidden_states: The collected hidden state vectors
        - equiv_classes: Ground truth equivalence class labels
        """
        sequences = self.generate_sequences(num_samples, max_len)

        equiv_classes = []
        hidden_states = []

        for seq in sequences:
            # Ground truth: which symbolic state does this reach?
            eq_class = self.mealy.get_state_after_sequence(seq)
            equiv_classes.append(eq_class)

            # SSM: what hidden state do we get?
            h = self.get_ssm_hidden_state(seq)
            hidden_states.append(h)

        equiv_classes = np.array(equiv_classes)
        hidden_states = np.array(hidden_states)

        # Statistics
        unique_classes = np.unique(equiv_classes)
        n_classes = len(unique_classes)

        print(f"Found {n_classes} equivalence classes (symbolic states)")
        class_counts = dict(zip(*np.unique(equiv_classes, return_counts=True)))
        print(f"Class distribution: {class_counts}")

        # Cluster hidden states with k-means
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        predicted_clusters = kmeans.fit_predict(hidden_states)

        # Clustering metrics
        ari = adjusted_rand_score(equiv_classes, predicted_clusters)
        nmi = normalized_mutual_info_score(equiv_classes, predicted_clusters)

        # Compute centroids and variances
        centroids = {}
        within_var = 0.0

        for cls in unique_classes:
            mask = equiv_classes == cls
            cls_points = hidden_states[mask]
            centroids[cls] = cls_points.mean(axis=0)
            within_var += np.sum((cls_points - centroids[cls]) ** 2)

        within_var /= len(hidden_states)

        # Between-class variance
        global_centroid = hidden_states.mean(axis=0)
        between_var = 0.0
        for cls in unique_classes:
            mask = equiv_classes == cls
            n_cls = mask.sum()
            between_var += n_cls * np.sum((centroids[cls] - global_centroid) ** 2)
        between_var /= len(hidden_states)

        # Pairwise centroid distances
        centroid_dists = []
        cls_list = list(unique_classes)
        for i in range(len(cls_list)):
            for j in range(i + 1, len(cls_list)):
                dist = np.linalg.norm(centroids[cls_list[i]] - centroids[cls_list[j]])
                centroid_dists.append((cls_list[i], cls_list[j], dist))

        results = {
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "n_equivalence_classes": n_classes,
            "within_class_variance": within_var,
            "between_class_variance": between_var,
            "separation_ratio": between_var / (within_var + 1e-8),
            "mean_centroid_distance": (
                np.mean([d[2] for d in centroid_dists]) if centroid_dists else 0
            ),
            "min_centroid_distance": (
                np.min([d[2] for d in centroid_dists]) if centroid_dists else 0
            ),
            "centroid_distances": centroid_dists,
            "hidden_states": hidden_states,
            "equiv_classes": equiv_classes,
            "centroids": centroids,
            "sequences": sequences,
            "predicted_clusters": predicted_clusters,
        }

        return results

    def visualize(self, results: Dict, save_path: str = "equivalence_analysis.png"):
        """Visualize hidden states colored by equivalence class."""
        hidden_states = results["hidden_states"]
        equiv_classes = results["equiv_classes"]
        centroids = results["centroids"]

        # PCA to 2D
        pca = PCA(n_components=2)
        hidden_2d = pca.fit_transform(hidden_states)

        # Project centroids
        centroid_matrix = np.array([centroids[c] for c in sorted(centroids.keys())])
        centroids_2d = pca.transform(centroid_matrix)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Hidden states colored by ground truth equivalence class
        scatter = axes[0].scatter(
            hidden_2d[:, 0],
            hidden_2d[:, 1],
            c=equiv_classes,
            cmap="tab10",
            alpha=0.6,
            s=30,
        )
        axes[0].scatter(
            centroids_2d[:, 0],
            centroids_2d[:, 1],
            c="black",
            marker="X",
            s=200,
            edgecolors="white",
            linewidths=2,
            label="Centroids",
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title(
            f"SSM Hidden States by Equivalence Class\n"
            f'ARI={results["adjusted_rand_index"]:.3f}, '
            f'NMI={results["normalized_mutual_info"]:.3f}'
        )
        axes[0].legend()
        plt.colorbar(scatter, ax=axes[0], label="Equivalence Class (State)")

        # Plot 2: Class distribution
        unique_classes, counts = np.unique(equiv_classes, return_counts=True)
        axes[1].bar(
            unique_classes.astype(str), counts, color="steelblue", edgecolor="black"
        )
        axes[1].set_xlabel("Equivalence Class (Symbolic State)")
        axes[1].set_ylabel("Sample Count")
        axes[1].set_title(
            f"Samples per Equivalence Class\n"
            f'Separation Ratio: {results["separation_ratio"]:.2f}'
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Visualization saved to {save_path}")

    def print_report(self, results: Dict):
        """Print analysis report."""
        print("\n" + "=" * 60)
        print("MYHILL-NERODE EQUIVALENCE ANALYSIS REPORT")
        print("=" * 60)

        print(f"\nAutomaton has {results['n_equivalence_classes']} equivalence classes")

        print("\nClustering Metrics:")
        print(f"  Adjusted Rand Index:      {results['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info:   {results['normalized_mutual_info']:.4f}")

        print("\nGeometric Separation:")
        print(f"  Within-class variance:    {results['within_class_variance']:.4f}")
        print(f"  Between-class variance:   {results['between_class_variance']:.4f}")
        print(f"  Separation ratio:         {results['separation_ratio']:.4f}")
        print(f"  Mean centroid distance:   {results['mean_centroid_distance']:.4f}")
        print(f"  Min centroid distance:    {results['min_centroid_distance']:.4f}")

        if results["centroid_distances"]:
            print("Pairwise centroid distances:")
            for c1, c2, dist in sorted(
                results["centroid_distances"], key=lambda x: x[2]
            ):
                print(f"  State {c1} <-> State {c2}: {dist:.4f}")

        print("\nInterpretation:")
        ari = results["adjusted_rand_index"]
        sep = results["separation_ratio"]

        if ari > 0.9:
            print("  ✓ Excellent: Hidden states strongly align with symbolic states")
        elif ari > 0.7:
            print("  ◐ Good: SSM learned approximate state structure")
        elif ari > 0.5:
            print("  ◔ Moderate: Some structure learned, but noisy")
        else:
            print("  ✗ Poor: Hidden states don't reflect symbolic structure")

        if sep > 10:
            print("  ✓ Equivalence classes are well-separated in hidden space")
        elif sep > 2:
            print("  ◐ Moderate separation between classes")
        else:
            print("  ✗ Classes overlap significantly in hidden space")

        print("=" * 60 + "\n")


def run_analysis(
    model_path: str,
    hoa_file: str,
    input_aps: List[str],
    output_aps: List[str],
    num_samples: int = 500,
    max_len: int = 10,
    device: str = "cpu",
):
    """Complete analysis pipeline.

    Args:
        model_path: Path to saved SSM model checkpoint
        hoa_file: Path to HOA file defining the Mealy machine
        input_aps: List of input AP names (must match HOA file)
        output_aps: List of output AP names (must match HOA file)
        num_samples: Number of random sequences to test
        max_len: Maximum sequence length
        device: 'cpu' or 'cuda'

    Returns:
        Results dictionary with all metrics and data
    """
    from State_Space_Model import FSM_SSM

    # Load trained model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = FSM_SSM(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        state_dim=checkpoint["state_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Parse Mealy machine from HOA
    print(f"Parsing Mealy machine from {hoa_file}")
    mealy = MealyMachine(hoa_file, input_aps, output_aps)
    mealy.print_transition_table()

    # Run equivalence analysis
    analyzer = EquivalenceAnalyzer(model, mealy, device)
    results = analyzer.analyze(num_samples=num_samples, max_len=max_len)

    # Output results
    analyzer.print_report(results)
    analyzer.visualize(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze SSM hidden states vs Myhill-Nerode equivalence classes"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model (.pt file)"
    )
    parser.add_argument("--hoa", type=str, required=True, help="Path to HOA file")
    parser.add_argument(
        "--inputs", type=str, required=True, help="Comma-separated input AP names"
    )
    parser.add_argument(
        "--outputs", type=str, required=True, help="Comma-separated output AP names"
    )
    parser.add_argument(
        "--samples", type=int, default=500, help="Number of test sequences"
    )
    parser.add_argument(
        "--max-len", type=int, default=10, help="Maximum sequence length"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")

    args = parser.parse_args()

    results = run_analysis(
        model_path=args.model,
        hoa_file=args.hoa,
        input_aps=args.inputs.split(","),
        output_aps=args.outputs.split(","),
        num_samples=args.samples,
        max_len=args.max_len,
        device=args.device,
    )
