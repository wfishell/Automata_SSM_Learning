#!/usr/bin/env python3
import sys, re, argparse
from typing import List, Dict, Tuple, Optional, Set

# ---------------- HOA parsing ----------------

class Transition:
    def __init__(self, target: int, label_str: str):
        self.target = target
        self.label_str = label_str  # e.g. '0&!1&2 | 0&!1&3'

class HOA:
    def __init__(self, ap: List[str], controllable: List[int], start: int,
                 states: Dict[int, List[Transition]]):
        self.ap = ap
        self.controllable = controllable
        self.start = start
        self.states = states  # state -> [Transition]

HOA_LABEL_RE = re.compile(r'^\[([^\]]+)\]\s+(\d+)$')

def parse_hoa_text(text: str) -> HOA:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    ap: List[str] = []
    controllable: List[int] = []
    start = 0
    states: Dict[int, List[Transition]] = {}
    current_state: Optional[int] = None
    for ln in lines:
        if ln.startswith("AP:"):
            ap = re.findall(r'"([^"]+)"', ln)
        elif ln.startswith("controllable-AP:"):
            controllable = list(map(int, re.findall(r'\d+', ln)))
        elif ln.startswith("Start:"):
            start = int(re.findall(r'\d+', ln)[0])
        elif ln.startswith("State:"):
            current_state = int(re.findall(r'\d+', ln)[0])
            states[current_state] = []
        elif ln.startswith("["):
            m = HOA_LABEL_RE.match(ln)
            if m and current_state is not None:
                states[current_state].append(Transition(int(m.group(2)), m.group(1).strip()))
    if not ap:
        raise SystemExit("HOA parse error: missing AP line.")
    if not states:
        raise SystemExit("HOA parse error: no states found.")
    return HOA(ap=ap, controllable=controllable, start=start, states=states)

# --------------- Label evaluation ----------------

def clause_satisfiable(clause: str, known: Dict[int, int], ctrl_free: Set[int]) -> bool:
    """
    clause like '0&!1&2' over AP indices; True if we can set ctrl_free vars to satisfy it.
    known = AP index -> value for inputs + any pinned outputs at this step
    ctrl_free = set(AP indices) the controller is allowed to choose this step
    """
    req_ctrl: Dict[int, int] = {}
    for tok in (t.strip() for t in clause.split("&") if t.strip()):
        neg = tok.startswith("!")
        idx = int(tok[1:]) if neg else int(tok)
        val = 0 if neg else 1
        if idx in known:
            if known[idx] != val:
                return False
        elif idx in ctrl_free:
            if idx in req_ctrl and req_ctrl[idx] != val:
                return False
            req_ctrl[idx] = val
        else:
            return False
    return True

def label_satisfiable(label: str, known: Dict[int,int], ctrl_free: Set[int]) -> bool:
    return any(clause_satisfiable(c.strip(), known, ctrl_free) for c in label.split("|"))

def step_next_state(hoa: HOA, state: int, known: Dict[int,int]) -> Optional[int]:
    ctrl_free = set(hoa.controllable) - set(known.keys())
    for tr in hoa.states.get(state, []):
        if label_satisfiable(tr.label_str, known, ctrl_free):
            return tr.target
    return None

# --------------- Compact trace parsing ----------------

def parse_compact_trace(trace_str: str,
                        ap_names: List[str],
                        controllable_idx: List[int],
                        cli_inputs: List[str],
                        cli_outputs: List[str]) -> Tuple[List[Dict[str,int]], List[Dict[str,int]]]:
    """
    Compact step format: <inputs...>,<outputs...>;
    If --inputs/--outputs omitted, defaults to env APs (non-controllable) then controllable APs, in HOA order.
    """
    ctrl_names = [ap_names[i] for i in controllable_idx]
    env_names  = [n for i,n in enumerate(ap_names) if i not in controllable_idx]

    inputs_syms  = cli_inputs  if cli_inputs  else env_names
    outputs_syms = cli_outputs if cli_outputs else ctrl_names

    steps_raw = [s.strip() for s in trace_str.strip().split(";") if s.strip()]
    if not steps_raw:
        raise SystemExit("Compact trace is empty.")

    width_expected = len(inputs_syms) + len(outputs_syms)
    inputs: List[Dict[str,int]] = []
    outputs: List[Dict[str,int]] = []

    for idx, step in enumerate(steps_raw):
        toks = [t.strip() for t in step.split(",")]
        if len(toks) != width_expected:
            raise SystemExit(
                f"Step {idx} width mismatch: got {len(toks)}, expected {width_expected} "
                f"({len(inputs_syms)} inputs + {len(outputs_syms)} outputs). Step: '{step}'"
            )
        if any(t not in ("0","1") for t in toks):
            raise SystemExit(f"Non-binary values at step {idx}: {toks}")

        in_vals  = list(map(int, toks[:len(inputs_syms)]))
        out_vals = list(map(int, toks[len(inputs_syms):]))
        inputs.append({n: v for n, v in zip(inputs_syms,  in_vals)})
        outputs.append({n: v for n, v in zip(outputs_syms, out_vals)})

    # sanity
    missing_env = [n for n in env_names if n not in inputs_syms]
    if missing_env:
        raise SystemExit(f"Compact trace missing required input APs: {missing_env}. Inputs={inputs_syms}")
    bad_outs = [n for n in outputs_syms if n not in ctrl_names]
    if bad_outs:
        raise SystemExit(f"Compact trace outputs {bad_outs} are not controllable. Controllables={ctrl_names}")
    return inputs, outputs

# --------------- Checking ----------------

def check_trace_against_controller(hoa: HOA,
                                   inputs: List[Dict[str,int]],
                                   outputs: List[Dict[str,int]]) -> Tuple[bool, Optional[int], Optional[str]]:
    ap_to_idx = {name: i for i, name in enumerate(hoa.ap)}
    ctrl_set = set(hoa.controllable)
    state = hoa.start
    T = len(inputs)
    if outputs and len(outputs) != T:
        return False, None, "length mismatch between inputs and outputs"

    for t in range(T):
        known: Dict[int,int] = {}
        for name, val in inputs[t].items():
            if name not in ap_to_idx:
                return False, t, f"Unknown AP in inputs at step {t}: {name}"
            known[ap_to_idx[name]] = int(val)
        if outputs:
            for name, val in outputs[t].items():
                if name not in ap_to_idx:
                    return False, t, f"Unknown AP in outputs at step {t}: {name}"
                known[ap_to_idx[name]] = int(val)

        nxt = step_next_state(hoa, state, known)
        if nxt is None:
            inv = {i:n for i,n in enumerate(hoa.ap)}
            pin_inputs  = {inv[k]: v for k, v in known.items() if k not in ctrl_set}
            pin_outputs = {inv[k]: v for k, v in known.items() if k in ctrl_set}
            return False, t, (f"No enabled transition from state {state} at step {t} "
                              f"for inputs={pin_inputs} outputs={pin_outputs}.")
        state = nxt
    return True, None, None

# --------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Check compact finite traces (one or many) against an HOA controller. "
                    "Compact trace step format: 'i1,...,o1,...;i1,...;...'"
    )
    ap.add_argument("hoa", help="controller .hoa")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--trace", help="Single compact trace string (quote it).")
    group.add_argument("--file", help="Path to a file with one compact trace per line.")
    ap.add_argument("--inputs", default="", help="Comma list of input AP names IN ORDER (defaults to env APs from HOA).")
    ap.add_argument("--outputs", default="", help="Comma list of output AP names IN ORDER (defaults to controllable APs).")
    args = ap.parse_args()

    with open(args.hoa, "r") as f:
        hoa = parse_hoa_text(f.read())

    cli_inputs  = [s.strip() for s in args.inputs.split(",") if s.strip()]
    cli_outputs = [s.strip() for s in args.outputs.split(",") if s.strip()]

    total = 0
    fails = 0

    def check_one(s: str, label: str):
        nonlocal total, fails
        total += 1
        try:
            ins, outs = parse_compact_trace(s, hoa.ap, hoa.controllable, cli_inputs, cli_outputs)
            ok, t, msg = check_trace_against_controller(hoa, ins, outs)
            if not ok:
                fails += 1
        except SystemExit:
            fails += 1

    if args.trace is not None:
        check_one(args.trace, "trace")
    else:
        with open(args.file, "r") as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                check_one(s, f"line {i}")

    if total == 0:
        print("No traces checked.")
    else:
        pct = 100.0 * fails / total
        print(f"Checked {total} traces. {fails} were invalid ({pct:.2f}%).")

    sys.exit(0)

if __name__ == "__main__":
    main()
