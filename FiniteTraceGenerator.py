import argparse, re, sys, itertools, random
from typing import Dict, List, Tuple, Optional

EDGE_RE = re.compile(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\]')
INIT_RE = re.compile(r'\bI\s*->\s*(\w+)')
LABEL_SPLIT_RE = re.compile(r'\s*/\s*')

#build 
TOK_ID = 'ID'; TOK_NOT='!'; TOK_AND='&'; TOK_OR='|'; TOK_LP='('; TOK_RP=')'

def tokenize_bool(expr: str):
    s = expr.replace("true", "1").replace("false", "0")
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in '!&|()':
            yield (c, c); i += 1
        elif c in '01':
            j = i+1
            yield (TOK_ID, s[i:j]); i = j
        elif c.isalpha() or c == '_':
            j = i+1
            while j < n and (s[j].isalnum() or s[j] == '_'):
                j += 1
            yield (TOK_ID, s[i:j]); i = j
        else:
            raise ValueError(f"Unexpected char {c!r} in expression {expr!r}")

class BoolParser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def eat(self, kind=None):
        tok = self.peek()
        if tok[0] is None:
            raise ValueError("Unexpected end of input")
        if kind and tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok}")
        self.pos += 1
        return tok

    def parse_or(self):
        node = self.parse_and()
        while self.peek()[0] == TOK_OR:
            self.eat(TOK_OR)
            rhs = self.parse_and()
            node = ('or', node, rhs)
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.peek()[0] == TOK_AND:
            self.eat(TOK_AND)
            rhs = self.parse_not()
            node = ('and', node, rhs)
        return node

    # not_expr := '!' not_expr | atom
    def parse_not(self):
        if self.peek()[0] == TOK_NOT:
            self.eat(TOK_NOT)
            node = self.parse_not()
            return ('not', node)
        return self.parse_atom()

    def parse_atom(self):
        k, v = self.peek()
        if k == TOK_ID:
            self.eat()
            return ('id', v)
        if k == TOK_LP:
            self.eat(TOK_LP)
            node = self.parse_or()
            self.eat(TOK_RP)
            return node
        raise ValueError(f"Expected ID or '(', got {k!r}")

def parse_bool_expr(s: str):
    p = BoolParser(tokenize_bool(s))
    ast = p.parse_or()
    if p.peek()[0] is not None:
        raise ValueError(f"Trailing tokens in {s!r}")
    return ast

def eval_bool_ast(ast, env: Dict[str, bool]) -> bool:
    t = ast[0]
    if t == 'id':
        name = ast[1]
        if name == '1': return True
        if name == '0': return False
        return bool(env.get(name, False))
    if t == 'not':
        return not eval_bool_ast(ast[1], env)
    if t == 'and':
        return eval_bool_ast(ast[1], env) and eval_bool_ast(ast[2], env)
    if t == 'or':
        return eval_bool_ast(ast[1], env) or eval_bool_ast(ast[2], env)
    raise ValueError(f"Unknown AST node {t}")

def vars_in_bool_ast(ast) -> set:
    t = ast[0]
    if t == 'id':
        v = ast[1]
        return set() if v in ('0','1') else {v}
    if t in ('not',):
        return vars_in_bool_ast(ast[1])
    if t in ('and','or'):
        return vars_in_bool_ast(ast[1]) | vars_in_bool_ast(ast[2])
    return set()

#LTLf parsing for --assume (supports G, F, X, U, !, &, |, parentheses) ---

LTL_TOK_ID='ID'; LTL_TOK_NOT='!'; LTL_TOK_AND='&'; LTL_TOK_OR='|'; LTL_TOK_LP='('; LTL_TOK_RP=')'
LTL_TOK_G='G'; LTL_TOK_F='F'; LTL_TOK_X='X'; LTL_TOK_U='U'

def tokenize_ltl(s: str):
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in '!&|()':
            yield (c, c); i += 1
        else:
            if c.isalpha() or c == '_':
                j = i+1
                while j < n and (s[j].isalnum() or s[j] == '_'):
                    j += 1
                word = s[i:j]
                if word == 'G': yield (LTL_TOK_G, word)
                elif word == 'F': yield (LTL_TOK_F, word)
                elif word == 'X': yield (LTL_TOK_X, word)
                elif word == 'U': yield (LTL_TOK_U, word)
                else: yield (LTL_TOK_ID, word)
                i = j
            else:
                raise ValueError(f"Unexpected char {c!r} in LTL {s!r}")

class LTLParser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def eat(self, kind=None):
        tok = self.peek()
        if tok[0] is None:
            raise ValueError("Unexpected end of input")
        if kind and tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok}")
        self.pos += 1
        return tok

    def parse_or(self):
        node = self.parse_and()
        while self.peek()[0] == LTL_TOK_OR:
            self.eat(LTL_TOK_OR)
            rhs = self.parse_and()
            node = ('or', node, rhs)
        return node

    def parse_and(self):
        node = self.parse_until()
        while self.peek()[0] == LTL_TOK_AND:
            self.eat(LTL_TOK_AND)
            rhs = self.parse_until()
            node = ('and', node, rhs)
        return node

    # until := unary ('U' unary)*
    def parse_until(self):
        node = self.parse_unary()
        while self.peek()[0] == LTL_TOK_U:
            self.eat(LTL_TOK_U)
            rhs = self.parse_unary()
            node = ('U', node, rhs)
        return node

    # unary := '!' unary | 'G' unary | 'F' unary | 'X' unary | atom
    def parse_unary(self):
        k, v = self.peek()
        if k in (LTL_TOK_NOT, LTL_TOK_G, LTL_TOK_F, LTL_TOK_X):
            self.eat(k)
            sub = self.parse_unary()
            if k == LTL_TOK_NOT: return ('not', sub)
            if k == LTL_TOK_G:   return ('G', sub)
            if k == LTL_TOK_F:   return ('F', sub)
            if k == LTL_TOK_X:   return ('X', sub)
        return self.parse_atom()

    def parse_atom(self):
        k, v = self.peek()
        if k == LTL_TOK_ID:
            self.eat()
            return ('id', v)
        if k == LTL_TOK_LP:
            self.eat(LTL_TOK_LP)
            node = self.parse_or()
            self.eat(LTL_TOK_RP)
            return node
        raise ValueError(f"Expected ID or '(', got {k!r}")

def parse_ltl(s: str):
    p = LTLParser(tokenize_ltl(s))
    ast = p.parse_or()
    if p.peek()[0] is not None:
        raise ValueError(f"Trailing tokens in LTL {s!r}")
    return ast

def eval_ltlf(ast, seq: List[Dict[str,int]], i: int, memo: Dict=None) -> bool:
    if memo is None:
        memo = {}
    key = (id(ast), i)
    if key in memo:
        return memo[key]
    t = ast[0]
    n = len(seq)
    if t == 'id':
        res = bool(seq[i].get(ast[1], 0)) if 0 <= i < n else False
    elif t == 'not':
        res = not eval_ltlf(ast[1], seq, i, memo)
    elif t == 'and':
        res = eval_ltlf(ast[1], seq, i, memo) and eval_ltlf(ast[2], seq, i, memo)
    elif t == 'or':
        res = eval_ltlf(ast[1], seq, i, memo) or eval_ltlf(ast[2], seq, i, memo)
    elif t == 'X':
        res = (i+1 < n) and eval_ltlf(ast[1], seq, i+1, memo)
    elif t == 'G':
        res = all(eval_ltlf(ast[1], seq, j, memo) for j in range(i, n))
    elif t == 'F':
        res = any(eval_ltlf(ast[1], seq, j, memo) for j in range(i, n))
    elif t == 'U':
        res = False
        for j in range(i, n):
            if eval_ltlf(ast[2], seq, j, memo):
                ok = True
                for k in range(i, j):
                    if not eval_ltlf(ast[1], seq, k, memo):
                        ok = False; break
                if ok:
                    res = True
                    break
    else:
        raise ValueError(f"Unknown LTL node {t}")
    memo[key] = res
    return res

def ltlf_holds(seq: List[Dict[str,int]], ast) -> bool:
    return eval_ltlf(ast, seq, 0, {})

def vars_in_ltl(ast) -> set:
    t = ast[0]
    if t == 'id':
        return {ast[1]}
    if t in ('not','G','F','X'):
        return vars_in_ltl(ast[1])
    if t in ('and','or','U'):
        return vars_in_ltl(ast[1]) | vars_in_ltl(ast[2])
    return set()

# --- DOT parsing ---

def load_transducer_from_dot(path: str):
    with open(path) as f:
        lines = f.readlines()

    start = None
    for line in lines:
        m0 = INIT_RE.search(line)
        if m0:
            start = m0.group(1)
            break

    transitions = {}  # state -> list of (input_ast, next_state, output_ast, raw_label)
    for line in lines:
        m = EDGE_RE.search(line)
        if not m:
            continue
        u, v, label = m.groups()
        parts = LABEL_SPLIT_RE.split(label, maxsplit=1)
        if len(parts) != 2:
            continue
        in_s, out_s = parts[0].strip(), parts[1].strip()
        in_ast  = parse_bool_expr(in_s)
        out_ast = parse_bool_expr(out_s)
        transitions.setdefault(u, []).append((in_ast, v, out_ast, label))
    if start is None:
        if transitions:
            start = next(iter(transitions.keys()))
        else:
            raise SystemExit("Could not detect initial state; use --start")
    return start, transitions

# --- Output valuation selection ---

def choose_output_valuation(out_ast, outputs: List[str], *, randomize=False, rng: Optional[random.Random]=None) -> Dict[str, int]:
    syms = outputs[:]
    sats = []
    for bits in itertools.product([0,1], repeat=len(syms)):
        env = {s: bool(b) for s, b in zip(syms, bits)}
        if eval_bool_ast(out_ast, env):
            sats.append({s: int(env[s]) for s in syms})
    if sats:
        if randomize:
            rng = rng or random
            return rng.choice(sats)
        return min(sats, key=lambda m: tuple(m[s] for s in syms))
    return {s: 0 for s in outputs}

# --- Utilities ---

def parse_stream(s: str) -> List[Dict[str,int]]:
    steps = []
    for chunk in filter(None, [t.strip() for t in s.split(';')]):
        d: Dict[str,int] = {}
        for kv in chunk.split(','):
            if not kv.strip():
                continue
            if '=' not in kv:
                raise SystemExit(f"Bad input token {kv!r}; expected key=value")
            k, v = kv.split('=')
            vv = v.strip().lower()
            if vv in ('1','true'): d[k.strip()] = 1
            elif vv in ('0','false'): d[k.strip()] = 0
            else: raise SystemExit(f"Bad value {v!r} for {k!r}; use 0/1/true/false")
        steps.append(d)
    return steps

def autodetect_ios(transitions) -> Tuple[List[str], List[str]]:
    in_vars, out_vars = set(), set()
    for _, edges in transitions.items():
        for in_ast, _, out_ast, _ in edges:
            in_vars  |= vars_in_bool_ast(in_ast)
            out_vars |= vars_in_bool_ast(out_ast)
    return sorted(in_vars - out_vars), sorted(out_vars)

# --- Simulation ---

def simulate(start: str, transitions, inputs: List[Dict[str,int]], outputs_syms: List[str],
             *, randomize_outputs=False, rng: Optional[random.Random]=None):
    s = start
    outs = []
    for t, env in enumerate(inputs):
        edges = transitions.get(s, [])
        taken = None
        for in_ast, dst, out_ast, raw in edges:
            if eval_bool_ast(in_ast, env):
                taken = (dst, out_ast, raw)
                break
        if taken is None:
            raise SystemExit(f"No matching edge from state {s!r} at step {t} for inputs {env}")
        dst, out_ast, _ = taken
        out_val = choose_output_valuation(out_ast, outputs_syms, randomize=randomize_outputs, rng=rng)
        outs.append(out_val)
        s = dst
    return outs

# --- Main ---

def main():
    ap = argparse.ArgumentParser(description="Simulate Mealy transducer from DOT with 'input / output' labels.")
    ap.add_argument("dot", help="Path to DOT file")
    ap.add_argument("--stream", default="", help='Input sequence, e.g. "a=1,b=0; a=0,b=1"')
    ap.add_argument("--inputs", default="", help='Comma list to pin input symbols (auto-detected if empty)')
    ap.add_argument("--outputs", default="", help='Comma list to pin output symbols (auto-detected if empty)')
    ap.add_argument("--start", default="", help="Initial state (auto-detected from I->state if empty)")
    ap.add_argument("--rand-steps", type=int, default=0, help="If >0, generate a random input stream with N steps (ignores --stream).")
    ap.add_argument("--prob", default="", help='Bernoulli probs per input, e.g. "a=0.2,b=0.8" (default 0.5 each).')
    ap.add_argument("--num-seqs", type=int, default=1, help="How many random sequences to generate (with --rand-steps).")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility.")
    ap.add_argument("--randomize-outputs", action="store_true", help="Pick a satisfying output valuation at random when there are many.")
    ap.add_argument("--assume", default="", help='LTLf constraint over inputs using G,F,X,U,!,&,| and parentheses (checked on streams; used for rejection sampling during generation).')
    ap.add_argument("--max-tries", type=int, default=2000, help="Max attempts to draw a random sequence that satisfies --assume.")
    args = ap.parse_args()

    start, transitions = load_transducer_from_dot(args.dot)
    if args.start:
        start = args.start

    auto_in, auto_out = autodetect_ios(transitions)
    inputs_syms  = [t.strip() for t in args.inputs.split(",") if t.strip()] or auto_in
    outputs_syms = [t.strip() for t in args.outputs.split(",") if t.strip()] or auto_out
    if not outputs_syms:
        sys.exit("Could not detect outputs; pass --outputs p0,p1,...")

    rng = random.Random(args.seed) if args.seed is not None else random

    # Parse assumption if provided
    assume_ast = None
    if args.assume:
        try:
            assume_ast = parse_ltl(args.assume)
        except Exception as e:
            sys.exit(f"Failed to parse --assume: {e}")
        # Ensure all vars in assumption are declared inputs
        used = vars_in_ltl(assume_ast)
        extra = used - set(inputs_syms)
        if extra:
            sys.exit(f"--assume uses symbols not in --inputs: {sorted(extra)}")

    def parse_probs(s: str, syms: List[str]) -> Dict[str, float]:
        probs = {k: 0.5 for k in syms}
        if not s:
            return probs
        for kv in s.split(','):
            if not kv.strip():
                continue
            if '=' not in kv:
                raise SystemExit(f"Bad prob token {kv!r}; expected key=prob")
            k, v = kv.split('=')
            k = k.strip(); p = float(v.strip())
            if not (0.0 <= p <= 1.0):
                raise SystemExit(f"Prob out of range for {k}: {p}")
            probs[k] = p
        return probs

    def gen_random_inputs(n: int, syms: List[str], probs: Dict[str, float], rng) -> List[Dict[str,int]]:
        steps = []
        for _ in range(n):
            steps.append({k: 1 if rng.random() < probs.get(k, 0.5) else 0 for k in syms})
        return steps

    # Build input sequences
    if args.rand_steps > 0:
        probs = parse_probs(args.prob, inputs_syms)
        sequences = []
        for _ in range(args.num_seqs):
            if assume_ast is None:
                seq = gen_random_inputs(args.rand_steps, inputs_syms, probs, rng)
            else:
                # rejection sampling
                seq = None
                for _try in range(args.max_tries):
                    cand = gen_random_inputs(args.rand_steps, inputs_syms, probs, rng)
                    if ltlf_holds(cand, assume_ast):
                        seq = cand; break
                if seq is None:
                    sys.exit(f"Could not satisfy --assume within {args.max_tries} tries. Try adjusting --prob or increase --max-tries.")
            sequences.append(seq)
    else:
        if not args.stream:
            sys.exit("Provide --stream or --rand-steps.")
        seq = parse_stream(args.stream)
        if assume_ast is not None and not ltlf_holds(seq, assume_ast):
            sys.exit("Provided --stream violates --assume.")
        sequences = [seq]

    # Emit I/O CSV blocks and simulate
    header = ",".join(outputs_syms)
    for idx, input_steps in enumerate(sequences, 1):
        if len(sequences) > 1:
            print(f"# sequence {idx}")
        print("# inputs")
        print(",".join(inputs_syms))
        for env in input_steps:
            print(",".join(str(env.get(sym, 0)) for sym in inputs_syms))
        print("# outputs")
        outs = simulate(start, transitions, input_steps, outputs_syms,
                        randomize_outputs=args.randomize_outputs, rng=rng)
        print(header)
        for o in outs:
            print(",".join(str(o[sym]) for sym in outputs_syms))

if __name__ == "__main__":
    main()
