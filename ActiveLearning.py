"""
Convert SPOT-generated automata (from TLSF) to AALpy-compatible oracles for active learning.

Usage:
    python spot_to_aalpy.py input.dot
"""

import re
import sys
from collections import defaultdict
from itertools import product
from aalpy.SULs import SUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWalkEqOracle, WMethodEqOracle
from aalpy.utils import save_automaton_to_file, visualize_automaton


class SpotAutomatonOracle(SUL):
    """
    Oracle wrapper for SPOT-generated automaton.
    Implements AALpy's SUL (System Under Learning) interface.
    """
    
    def __init__(self, dot_file_path):
        super().__init__()
        self.dot_file = dot_file_path
        self.automaton = self._parse_dot_file(dot_file_path)
        self.propositions = self._extract_propositions()
        self.alphabet = self._generate_alphabet()
        self.current_state = self.automaton['initial']
        
        # Create transition lookup table for efficiency
        self.transition_map = self._build_transition_map()
        
        print(f"Loaded automaton from {dot_file_path}")
        print(f"  States: {len(self.automaton['states'])}")
        print(f"  Propositions: {self.propositions}")
        print(f"  Alphabet size: {len(self.alphabet)}")
    
    def _parse_dot_file(self, filepath):
        """Parse SPOT-generated DOT file."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract states
        states = set()
        state_pattern = r'^\s*(\d+)\s+\[label="(\d+)"\]'
        for match in re.finditer(state_pattern, content, re.MULTILINE):
            states.add(match.group(1))
        
        # Extract initial state
        initial_pattern = r'I\s+->\s+(\d+)'
        initial_match = re.search(initial_pattern, content)
        initial_state = initial_match.group(1) if initial_match else '0'
        
        # Extract transitions
        transitions = defaultdict(list)
        # Pattern for transitions (handles acceptance marks)
        trans_pattern = r'^\s*(\d+)\s+->\s+(\d+)\s+\[label="([^"]+)"[^\]]*\]'
        
        for match in re.finditer(trans_pattern, content, re.MULTILINE):
            from_state = match.group(1)
            to_state = match.group(2)
            # Remove acceptance marks like \n{0} or \n{1}
            condition = match.group(3).split('\n')[0].strip()
            
            transitions[from_state].append({
                'to': to_state,
                'condition': condition
            })
        
        return {
            'states': sorted(list(states)),
            'initial': initial_state,
            'transitions': dict(transitions)
        }
    
    def _extract_propositions(self):
        """Extract atomic propositions from transition conditions."""
        props = set()
        for trans_list in self.automaton['transitions'].values():
            for trans in trans_list:
                # Remove operators and parentheses
                cleaned = re.sub(r'[&|!()]', ' ', trans['condition'])
                # Find all proposition names
                found = re.findall(r'\b([a-zA-Z][a-zA-Z0-9]*)\b', cleaned)
                props.update(found)
        return sorted(list(props))
    
    def _generate_alphabet(self):
        """Generate alphabet as binary encodings of proposition valuations."""
        n = len(self.propositions)
        if n > 10:  # Warn if alphabet is very large
            print(f"Warning: {2**n} symbols in alphabet (may be slow)")
        
        alphabet = []
        for values in product([False, True], repeat=n):
            # Create binary string representation
            symbol = ''.join('1' if v else '0' for v in values)
            alphabet.append(symbol)
        return alphabet
    
    def _build_transition_map(self):
        """Pre-compute transition map for all state-input pairs."""
        trans_map = {}
        
        for state in self.automaton['states']:
            for symbol in self.alphabet:
                # Find which transition matches this input
                valuation = self._symbol_to_valuation(symbol)
                next_state = self._find_next_state(state, valuation)
                trans_map[(state, symbol)] = next_state
        
        return trans_map
    
    def _symbol_to_valuation(self, symbol):
        """Convert binary string to proposition valuation."""
        return {prop: symbol[i] == '1' 
                for i, prop in enumerate(self.propositions)}
    
    def _evaluate_condition(self, condition, valuation):
        """Evaluate boolean condition with given valuation."""
        expr = condition
        
        # Replace each proposition with its truth value
        for prop, value in valuation.items():
            # Use word boundaries to avoid partial matches
            expr = re.sub(r'\b' + prop + r'\b', str(value), expr)
        
        # Convert to Python boolean operators
        expr = expr.replace('&', ' and ')
        expr = expr.replace('|', ' or ')
        expr = expr.replace('!', ' not ')
        
        try:
            return eval(expr)
        except:
            print(f"Error evaluating: {condition} with {valuation}")
            return False
    
    def _find_next_state(self, state, valuation):
        """Find next state given current state and input valuation."""
        transitions = self.automaton['transitions'].get(state, [])
        
        for trans in transitions:
            if self._evaluate_condition(trans['condition'], valuation):
                return trans['to']
        
        # If no transition matches, stay in current state (or could go to error)
        return state
    
    def step(self, letter):
        """Execute one step of the automaton (AALpy interface)."""
        if letter not in self.alphabet:
            raise ValueError(f"Invalid input: {letter}")
        
        # Use precomputed transition map
        next_state = self.transition_map.get((self.current_state, letter), self.current_state)
        self.current_state = next_state
        
        # Return state as output (for Mealy machine interpretation)
        return next_state
    
    def reset(self):
        """Reset to initial state (AALpy interface)."""
        self.current_state = self.automaton['initial']
    
    def pre(self):
        """Called before learning (AALpy interface)."""
        self.reset()
    
    def post(self):
        """Called after learning (AALpy interface)."""
        pass


def learn_automaton_from_spot(dot_file, 
                              learning_algorithm='lstar',
                              eq_oracle_type='random_walk',
                              max_rounds=100):
    """
    Main function to learn an automaton using SPOT automaton as oracle.
    
    Args:
        dot_file: Path to SPOT-generated DOT file
        learning_algorithm: 'lstar' or 'kv'
        eq_oracle_type: 'random_walk' or 'w_method'
        max_rounds: Maximum learning rounds
    
    Returns:
        Learned AALpy automaton
    """
    # Create oracle from SPOT automaton
    oracle = SpotAutomatonOracle(dot_file)
    
    # Warn about alphabet size
    if len(oracle.alphabet) > 8 and eq_oracle_type == 'w_method':
        print(f"WARNING: W-method with {len(oracle.alphabet)} symbols may be very slow!")
        print("Consider using 'random_walk' instead.")
    
    # Setup equivalence oracle
    if eq_oracle_type == 'random_walk':
        # Scale parameters based on problem size
        num_steps = min(50000, len(oracle.alphabet) * 5000)
        
        from aalpy.oracles import RandomWalkEqOracle
        eq_oracle = RandomWalkEqOracle(
            alphabet=oracle.alphabet,
            sul=oracle,
            num_steps=num_steps,
            reset_prob=0.09,
            reset_after_cex=True  # Reset after finding counterexample
        )
        print(f"Using RandomWalk with {num_steps} steps")
        
    elif eq_oracle_type == 'w_method':
        from aalpy.oracles import WMethodEqOracle
        # Limit depth for large alphabets
        max_states = len(oracle.automaton['states']) * 2
        if len(oracle.alphabet) > 8:
            max_states = min(max_states, 6)
            
        eq_oracle = WMethodEqOracle(
            alphabet=oracle.alphabet,
            sul=oracle,
            max_number_of_states=max_states
        )
        print(f"Using W-method with max_states={max_states}")
    else:
        raise ValueError(f"Unknown equivalence oracle type: {eq_oracle_type}")
    
    # Run learning algorithm
    print(f"\nStarting {learning_algorithm.upper()} learning with {eq_oracle_type} equivalence oracle...")
    
    if learning_algorithm == 'lstar':
        from aalpy.learning_algs import run_Lstar
        learned = run_Lstar(
            alphabet=oracle.alphabet,
            sul=oracle,
            eq_oracle=eq_oracle,
            automaton_type='mealy',
            max_learning_rounds=max_rounds,
            print_level=2,
            cache_and_non_det_check=True  # Enable caching
        )
    elif learning_algorithm == 'kv':
        from aalpy.learning_algs import run_KV
        learned = run_KV(
            alphabet=oracle.alphabet,
            sul=oracle,
            eq_oracle=eq_oracle,
            automaton_type='mealy',
            max_learning_rounds=max_rounds,
            print_level=2
        )
    else:
        raise ValueError(f"Unknown learning algorithm: {learning_algorithm}")
    
    # Check if learning was complete
    if len(learned.states) < len(oracle.automaton['states']):
        print(f"\nWARNING: Learned only {len(learned.states)} states out of {len(oracle.automaton['states'])} oracle states.")
        print("The equivalence oracle may not have found all counterexamples.")
        print("Try: 1) Increasing num_steps for random_walk")
        print("     2) Using w_method for complete checking (if alphabet is small)")
        print("     3) Running learning multiple times")
    
    return learned, oracle


def compare_automata(learned, oracle):
    """Compare learned automaton with oracle on test sequences."""
    import random
    
    print("\n=== Comparing Learned vs Oracle ===")
    
    # Generate test sequences
    num_tests = 1000
    max_length = 20
    mismatches = 0
    example_mismatches = []
    
    for _ in range(num_tests):
        # Generate random sequence
        length = random.randint(1, max_length)
        sequence = [random.choice(oracle.alphabet) for _ in range(length)]
        
        # Execute on oracle
        oracle.reset()
        oracle_outputs = []
        for symbol in sequence:
            output = oracle.step(symbol)
            oracle_outputs.append(output)
        
        # Execute on learned automaton (direct traversal)
        learned_outputs = []
        current_state = learned.initial_state
        for symbol in sequence:
            if symbol in current_state.transitions:
                next_state = current_state.transitions[symbol]
                # For Mealy machines, get output from output_fun
                if hasattr(current_state, 'output_fun') and symbol in current_state.output_fun:
                    output = current_state.output_fun[symbol]
                else:
                    output = next_state.state_id
                learned_outputs.append(output)
                current_state = next_state
            else:
                # No transition defined - this shouldn't happen if learning was complete
                learned_outputs.append(None)
                break
        
        # Compare
        if oracle_outputs != learned_outputs:
            mismatches += 1
            if len(example_mismatches) < 3:  # Store first few mismatches
                example_mismatches.append({
                    'sequence': sequence[:5],
                    'oracle': oracle_outputs[:5],
                    'learned': learned_outputs[:5]
                })
    
    # Show example mismatches
    if example_mismatches:
        print("Example mismatches:")
        for ex in example_mismatches:
            print(f"  Seq {ex['sequence']}... Oracle: {ex['oracle']}..., Learned: {ex['learned']}...")
    
    accuracy = (num_tests - mismatches) / num_tests * 100
    print(f"Accuracy: {accuracy:.2f}% ({num_tests - mismatches}/{num_tests} sequences match)")
    
    return accuracy


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python spot_to_aalpy.py <dot_file> [algorithm] [eq_oracle]")
        print("  algorithm: lstar (default) or kv")
        print("  eq_oracle: random_walk (default) or w_method")
        sys.exit(1)
    
    dot_file = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else 'lstar'
    eq_oracle = sys.argv[3] if len(sys.argv) > 3 else 'random_walk'
    
    # Learn automaton
    learned, oracle = learn_automaton_from_spot(dot_file, algorithm, eq_oracle)
    
    # Print results
    print(f"\n=== Learning Complete ===")
    print(f"Oracle states: {len(oracle.automaton['states'])}")
    print(f"Learned states: {len(learned.states)}")
    
    # Save learned automaton
    output_file = dot_file.replace('.dot', '_learned.dot')
    save_automaton_to_file(learned, output_file)
    print(f"Learned automaton saved to: {output_file}")
    
    # Optional: Compare accuracy
    if len(oracle.automaton['states']) <= 10:  # Only for small automata
        compare_automata(learned, oracle)
    
    # Optional: Visualize (requires graphviz)
    try:
        visualize_automaton(learned, path=output_file.replace('.dot', ''))
        print(f"Visualization saved to: {output_file.replace('.dot', '.pdf')}")
    except:
        print("Visualization skipped (install graphviz to enable)")


if __name__ == '__main__':
    main()