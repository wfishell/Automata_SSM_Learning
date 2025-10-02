#!/usr/bin/env python3
"""
Learn a Mealy machine with RPNI from positive IO traces.

USAGE (positional args as requested):
    python passive_learning.py "a,b" "p0,p1" traces.txt --dump-dot learned_mealy.dot --dump-hoa learned_mealy.hoa

Trace file format (one trace per line):
    Each step has all input bits followed by all output bits, separated by commas.
    Steps are separated by semicolons ';'.

Example (for inputs a,b and outputs p0,p1 → 4 bits per step):
    0,0,0,1; 1,0,0,1; 0,0,0,1; 1,0,0,1; 1,1,1,0;

This script uses ONLY positive traces (no negatives) and classic RPNI.
"""

import argparse, sys
from typing import List, Tuple

# --- Import run_RPNI across AALPy versions ---
try:
    from aalpy.learning_algs.deterministic_passive.RPNI import run_RPNI
except Exception:
    import importlib
    run_RPNI = getattr(
        importlib.import_module('aalpy.learning_algs.deterministic_passive.RPNI'),
        'run_RPNI'
    )

def convert_i_o_traces_for_RPNI(
    io_traces: List[Tuple[List[str], List[str]]]
) -> List[Tuple[Tuple[str, ...], str]]:
    """
    Input: list of traces, each as (input_symbol_list, output_symbol_list), same length per trace.
    Output: prefix-closed list of (input_prefix_tuple, output_symbol_at_that_step).
    
    For Mealy machine learning: input sequences map to output symbols.
    """
    dataset: List[Tuple[Tuple[str, ...], str]] = []
    
    for trace_idx, (inp_seq, out_seq) in enumerate(io_traces):
        if len(inp_seq) != len(out_seq):
            raise ValueError(f"Trace {trace_idx}: Input/output lengths differ ({len(inp_seq)} vs {len(out_seq)}).")
        
        # Add all prefixes of this trace
        for t in range(len(inp_seq)):
            # Input sequence up to step t (inclusive)
            input_prefix = tuple(inp_seq[:t+1])
            # Output symbol at step t
            output_symbol = out_seq[t]
            
            dataset.append((input_prefix, output_symbol))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dataset = []
    for item in dataset:
        if item not in seen:
            seen.add(item)
            unique_dataset.append(item)
    
    print(f"Created dataset with {len(unique_dataset)} unique prefix-output pairs")
    print('this is the DATASET')
    print(uni)
    return unique_dataset

# --- Save DOT compat ---
def save_dot(model, path: str):
    """Save DOT file with better Spot compatibility."""
    if not path:
        return
    
    try:
        # Try AALPy's built-in methods first
        if hasattr(model, "save_to_file"):
            model.save_to_file(path)
            print(f"DOT saved via save_to_file to: {path}")
            return
        elif hasattr(model, "to_dot"):
            with open(path, "w") as f:
                f.write(model.to_dot())
            print(f"DOT saved via to_dot to: {path}")
            return
    except Exception as e:
        print(f"AALPy methods failed: {e}")
    
    # Try fallback via utils
    try:
        from aalpy.utils import save_automaton_to_file
        save_automaton_to_file(model, path)
        print(f"DOT saved via utils to: {path}")
        return
    except Exception as e:
        print(f"Utils method failed: {e}")
    
    # Manual DOT generation as last resort
    try:
        write_spot_compatible_dot(model, path)
        print(f"DOT saved via manual generation to: {path}")
    except Exception as e:
        print(f"Manual DOT generation failed: {e}")

def write_spot_compatible_dot(model, path: str):
    """Generate a Spot-compatible DOT file for a Mealy machine."""
    with open(path, "w") as f:
        f.write('digraph "" {\n')
        f.write('   rankdir=LR\n')
        f.write('   node [shape="circle"]\n')
        f.write('   I [label="", style=invis, width=0]\n')
        
        # Create state mapping
        state_to_id = {state: i for i, state in enumerate(model.states)}
        
        # Mark initial state
        initial_id = state_to_id[model.initial_state]
        f.write(f'   I -> {initial_id}\n')
        
        # Write states
        for state_id, state in enumerate(model.states):
            f.write(f'   {state_id} [label="{state_id}"]\n')
        
        # Write transitions in Mealy format: input / output
        for state_id, state in enumerate(model.states):
            if hasattr(state, 'transitions') and isinstance(state.transitions, dict):
                # Group transitions by target state to combine input conditions
                target_groups = {}
                
                for input_symbol, target_state in state.transitions.items():
                    target_id = state_to_id[target_state]
                    
                    # Get output for this input
                    output_symbol = ""
                    if hasattr(state, 'output_fun') and isinstance(state.output_fun, dict):
                        output_symbol = state.output_fun.get(input_symbol, "")
                    
                    # Convert input/output assignments to boolean expressions
                    input_expr = assignment_to_boolean_expr(input_symbol)
                    output_expr = assignment_to_boolean_expr(output_symbol)
                    
                    # Group by target and output
                    key = (target_id, output_expr)
                    if key not in target_groups:
                        target_groups[key] = []
                    target_groups[key].append(input_expr)
                
                # Write combined transitions in Mealy format
                for (target_id, output_expr), input_exprs in target_groups.items():
                    # Combine input expressions with OR
                    combined_input = " | ".join(f"({expr})" for expr in input_exprs)
                    if len(input_exprs) == 1:
                        combined_input = input_exprs[0]  # No need for parentheses
                    
                    # Mealy machine format: input / output
                    label = f"{combined_input} / {output_expr}"
                    f.write(f'   {state_id} -> {target_id} [label="{label}"]\n')
        
        f.write('}\n')

def assignment_to_boolean_expr(assignment_str: str) -> str:
    """Convert 'a=0,b=1,p0=1,p1=0' to '!a & b & p0 & !p1' format."""
    if not assignment_str:
        return "true"
    
    parts = assignment_str.split(',')
    conditions = []
    
    for part in parts:
        if '=' in part:
            var, val = part.split('=')
            var = var.strip()
            val = val.strip()
            
            if val == '1':
                conditions.append(var)
            elif val == '0':
                conditions.append(f"!{var}")
            # Skip invalid values
    
    if not conditions:
        return "true"
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return " & ".join(conditions)

# --- Save HOA format ---
def save_hoa(model, path: str, input_names: List[str], output_names: List[str]):
    """Save the learned Mealy machine in reactive synthesis HOA format."""
    if not path:
        return
    
    try:
        with open(path, "w") as f:
            # HOA header
            f.write("HOA: v1\n")
            f.write("name: \"Learned Mealy Machine\"\n")
            f.write("tool: \"RPNI Learning\"\n")
            f.write(f"States: {len(model.states)}\n")
            
            # All atomic propositions (inputs + outputs)
            all_aps = input_names + output_names
            f.write(f"AP: {len(all_aps)}")
            for name in all_aps:
                f.write(f' "{name}"')
            f.write("\n")
            
            # Controllable APs (outputs) - indices in the AP list
            output_indices = []
            for i, name in enumerate(all_aps):
                if name in output_names:
                    output_indices.append(str(i))
            f.write(f"controllable-AP: {' '.join(output_indices)}\n")
            
            # Start state
            initial_state_id = model.states.index(model.initial_state)
            f.write(f"Start: {initial_state_id}\n")
            
            # Acceptance condition (all states accepting)
            f.write("acc-name: all\n")
            f.write("Acceptance: 0 t\n")
            
            # Properties for reactive synthesis
            f.write("properties: deterministic\n")
            f.write("--BODY--\n")
            
            # Create state mapping
            state_to_id = {state: i for i, state in enumerate(model.states)}
            
            # Process each state
            for state_id, state in enumerate(model.states):
                f.write(f"State: {state_id}\n")
                
                if hasattr(state, 'transitions') and isinstance(state.transitions, dict):
                    # For each input, create a transition with the full AP valuation
                    
                    for input_symbol, target_state in state.transitions.items():
                        target_id = state_to_id[target_state]
                        
                        # Get output for this input
                        output_symbol = ""
                        if hasattr(state, 'output_fun') and isinstance(state.output_fun, dict):
                            output_symbol = state.output_fun.get(input_symbol, "")
                        
                        # Create full AP valuation (inputs + outputs)
                        input_vals = parse_symbol_to_bits(input_symbol, input_names)
                        output_vals = parse_symbol_to_bits(output_symbol, output_names)
                        
                        # Combine into full AP valuation
                        all_vals = input_vals + output_vals
                        
                        # Create boolean condition for all APs
                        ap_conditions = []
                        for i, val in enumerate(all_vals):
                            if val == 1:
                                ap_conditions.append(str(i))
                            else:
                                ap_conditions.append(f"!{i}")
                        
                        ap_condition = "&".join(ap_conditions) if ap_conditions else "t"
                        
                        # Write transition with full AP condition
                        f.write(f"[{ap_condition}] {target_id}\n")
            
            f.write("--END--\n")
        print(f"HOA saved to: {path}")
    
    except Exception as e:
        print(f"Warning: Could not save HOA file: {e}")
        import traceback
        traceback.print_exc()

def parse_combined_symbol_to_bits(symbol: str, all_ap_names: List[str]) -> List[int]:
    """Parse a combined symbol like 'a=0,b=1,p0=1,p1=0' to list of bits for all APs."""
    bits = []
    pairs = symbol.split(',')
    name_to_val = {}
    
    for pair in pairs:
        if '=' in pair:
            name, val = pair.split('=')
            name_to_val[name.strip()] = int(val.strip())
    
    for name in all_ap_names:
        bits.append(name_to_val.get(name, 0))
    
    return bits

def parse_symbol_to_bits(symbol: str, var_names: List[str]) -> List[int]:
    """Parse a symbol like 'a=0,b=1' to list of bits [0, 1]."""
    bits = []
    pairs = symbol.split(',')
    name_to_val = {}
    
    for pair in pairs:
        name, val = pair.split('=')
        name_to_val[name] = int(val)
    
    for name in var_names:
        bits.append(name_to_val.get(name, 0))
    
    return bits

# --- Parse one line of compact IO trace ---
def parse_trace_line(
    line: str,
    input_names: List[str],
    output_names: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Line format (one trace per line):
      <in...,out...> ; <in...,out...> ; ...
    Bits are 0/1. Whitespaces are allowed.
    Returns: (list of input SYMBOLS, list of output SYMBOLS);
             each symbol packs the whole valuation for that step as a single token string.
    """
    steps = [s.strip() for s in line.strip().split(";") if s.strip()]
    in_tokens, out_tokens = [], []
    in_w = len(input_names); out_w = len(output_names)

    for idx, step in enumerate(steps):
        parts = [p.strip() for p in step.split(",") if p.strip() != ""]
        expected = in_w + out_w
        if len(parts) != expected:
            raise ValueError(
                f"Step {idx} has {len(parts)} values, expected {expected} "
                f"({in_w} inputs + {out_w} outputs). Step='{step}'"
            )
        if any(p not in ("0","1") for p in parts):
            raise ValueError(f"Non-binary entry at step {idx}: {parts}")

        in_bits  = list(map(int, parts[:in_w]))
        out_bits = list(map(int, parts[in_w:]))

        # Pack entire valuation per step into one symbol string (OK for AALPy)
        in_tok  = ",".join(f"{n}={b}" for n, b in zip(input_names, in_bits))
        out_tok = ",".join(f"{n}={b}" for n, b in zip(output_names, out_bits))

        in_tokens.append(in_tok)
        out_tokens.append(out_tok)

    return in_tokens, out_tokens

def read_traces_file(
    path: str,
    input_names: List[str],
    output_names: List[str],
) -> List[Tuple[List[str], List[str]]]:
    traces = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            inp, out = parse_trace_line(s, input_names, output_names)
            if len(inp) != len(out):
                raise ValueError(f"Line {i}: input/output lengths differ ({len(inp)} vs {len(out)}).")
            traces.append((inp, out))
    return traces

# --- Runner for Mealy machine learning ---
def run_rpni_mealy_positive(pos_pairs):
    """
    Build prefix-closed dataset and call AALPy RPNI for a Mealy machine.
    """
    import inspect

    dataset = convert_i_o_traces_for_RPNI(pos_pairs)
    sig = inspect.signature(run_RPNI)
    params = list(sig.parameters.keys())
    
    print(f"Available RPNI parameters: {params}")
    print(f"Dataset sample (first 3): {dataset[:3]}")

    # Learn as a Mealy machine (the correct approach for input-output traces)
    try:
        # Your specific AALPy version signature - using 'mealy' for input-output behavior
        result = run_RPNI(
            data=dataset,
            automaton_type='mealy',
            algorithm='classic',
            input_completeness=True,
            print_info=True
        )
        return result
    except Exception as e1:
        print(f"Specific signature attempt failed: {e1}")
        
    try:
        # Try with minimal required parameters
        result = run_RPNI(
            data=dataset,
            automaton_type='mealy'
        )
        return result
    except Exception as e2:
        print(f"Minimal parameters attempt failed: {e2}")
        
    try:
        # Try positional arguments
        result = run_RPNI(dataset, 'mealy', 'classic')
        return result
    except Exception as e3:
        print(f"Positional arguments attempt failed: {e3}")
        
    try:
        # Try with just data and automaton_type positionally
        result = run_RPNI(dataset, 'mealy')
        return result
    except Exception as e4:
        print(f"Two positional args attempt failed: {e4}")
        
    # If all fails, let's inspect what run_RPNI expects more carefully
    print(f"Full signature: {sig}")
    for param_name, param in sig.parameters.items():
        print(f"  {param_name}: default={param.default}, annotation={param.annotation}")
    
    raise RuntimeError(f"Could not determine correct run_RPNI signature. Available params: {params}")


def main():
    ap = argparse.ArgumentParser(
        description="Learn a Mealy machine via classic RPNI from positive IO traces."
    )
    # As requested: three positional arguments
    ap.add_argument("inputs", help="Comma list of input AP names, e.g. 'a,b'")
    ap.add_argument("outputs", help="Comma list of output AP names, e.g. 'p0,p1'")
    ap.add_argument("traces", help="Path to trace file (one trace per line).")
    ap.add_argument("--dump-dot", default="", help="Optional path to save the learned Mealy as DOT.")
    ap.add_argument("--dump-hoa", default="", help="Optional path to save the learned Mealy as HOA.")
    args = ap.parse_args()

    input_names  = [s.strip() for s in args.inputs.split(",")  if s.strip()]
    output_names = [s.strip() for s in args.outputs.split(",") if s.strip()]
    if not input_names or not output_names:
        print("Error: inputs and outputs must be non-empty.", file=sys.stderr)
        sys.exit(2)

    pos_pairs = read_traces_file(args.traces, input_names, output_names)
    if not pos_pairs:
        print("Error: no positive traces loaded.", file=sys.stderr)
        sys.exit(2)

    print(f"Loaded {len(pos_pairs)} positive traces")
    mealy = run_rpni_mealy_positive(pos_pairs)

    # Report
    if hasattr(mealy, "get_input_alphabet"):
        sigma_size = len(mealy.get_input_alphabet())
    else:
        sigma_size = len({a for (inp, _out) in pos_pairs for a in inp})
    
    print(f"Learned Mealy Machine: |Q|={len(mealy.states)}  |Σ|={sigma_size}  (+:{len(pos_pairs)})")

    # Save DOT
    if args.dump_dot:
        save_dot(mealy, args.dump_dot)
        print(f"DOT saved to: {args.dump_dot}")
    
    # Save HOA
    if args.dump_hoa:
        save_hoa(mealy, args.dump_hoa, input_names, output_names)
    
    # Print DOT if no file specified
    if not args.dump_dot and not args.dump_hoa:
        if hasattr(mealy, "to_dot"):
            print(mealy.to_dot())
        else:
            print("(No DOT available on this AALPy version; use --dump-dot to save via utils.)")

if __name__ == "__main__":
    main()