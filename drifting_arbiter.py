"""Drifting Arbiter Trace Generator.

Generates traces that:
- Initially behave exactly like the HOA arbiter
- Gradually drift to "favor the leader" (whoever has more grants)
- Always respect starvation protection (5-step limit)

AP mapping from HOA:
  AP 0: g_0 (controllable)
  AP 1: c_0 (environment)
  AP 2: r_0 (environment)
  AP 3: g_1 (controllable)
  AP 4: c_1 (environment)
  AP 5: r_1 (environment)
"""

import random
from dataclasses import dataclass


@dataclass
class Step:
    """One step in the trace."""

    timestep: int
    r_0: bool
    r_1: bool
    c_0: bool
    c_1: bool
    g_0: bool
    g_1: bool
    state: int
    next_state: int
    count_0: int  # cumulative grants to g_0
    count_1: int  # cumulative grants to g_1
    pending_0: int  # how long g_0 has been waiting
    pending_1: int  # how long g_1 has been waiting
    drift_weight: float  # probability of favoring leader
    used_drift: bool  # did we use drift logic this step?


class OriginalHOAController:
    """The original deterministic controller from the HOA.

    Returns (g_0, g_1, next_state) for given (state, c_0, r_0, c_1, r_1).
    """

    def __init__(self):
        self.transitions = self._build_transitions()

    def _build_transitions(self):
        """Build transition table from HOA.

        State: 0
        [!0&1&!3&4 | !0&1&!3&!5 | !0&!2&!3&4 | !0&!2&!3&!5] 0
        [!0&1&!3&!4&5 | !0&!2&!3&!4&5] 1
        [!0&!1&2&!3&4 | !0&!1&2&!3&!5] 2
        [0&!1&2&!3&!4&5] 3

        AP: 0=g_0, 1=c_0, 2=r_0, 3=g_1, 4=c_1, 5=r_1
        """
        transitions = {}

        # State 0
        transitions[0] = [
            # [!0&1&!3&4 | !0&1&!3&!5 | !0&!2&!3&4 | !0&!2&!3&!5] -> 0
            # c_0 or !r_0, and c_1 or !r_1 => no grant
            (lambda c0, r0, c1, r1: (c0 or not r0) and (c1 or not r1), False, False, 0),
            # [!0&1&!3&!4&5 | !0&!2&!3&!4&5] -> 1
            # (c_0 or !r_0), and !c_1 and r_1 => no grant, but r_1 pending
            (
                lambda c0, r0, c1, r1: (c0 or not r0) and (not c1 and r1),
                False,
                False,
                1,
            ),
            # [!0&!1&2&!3&4 | !0&!1&2&!3&!5] -> 2
            # !c_0 and r_0, and (c_1 or !r_1) => no grant, but r_0 pending
            (
                lambda c0, r0, c1, r1: (not c0 and r0) and (c1 or not r1),
                False,
                False,
                2,
            ),
            # [0&!1&2&!3&!4&5] -> 3
            # !c_0 and r_0 and !c_1 and r_1 => grant g_0
            (
                lambda c0, r0, c1, r1: (not c0 and r0) and (not c1 and r1),
                True,
                False,
                3,
            ),
        ]

        # State 1: r_1 was pending
        transitions[1] = [
            # [!0&1&!3&4 | !0&!2&!3&4] -> 0
            # (c_0 or !r_0) and c_1 => no grant
            (lambda c0, r0, c1, r1: (c0 or not r0) and c1, False, False, 0),
            # [!0&!1&2&!3&4] -> 2
            # !c_0 and r_0 and c_1 => no grant
            (lambda c0, r0, c1, r1: (not c0 and r0) and c1, False, False, 2),
            # [!0&1&3&!4 | !0&!2&3&!4] -> 0
            # (c_0 or !r_0) and !c_1 => grant g_1
            (lambda c0, r0, c1, r1: (c0 or not r0) and not c1, False, True, 0),
            # [!0&!1&2&3&!4] -> 4
            # !c_0 and r_0 and !c_1 => grant g_1
            (lambda c0, r0, c1, r1: (not c0 and r0) and not c1, False, True, 4),
        ]

        # State 2: r_0 was pending
        transitions[2] = [
            # [!0&1&!3&4 | !0&1&!3&!5] -> 0
            # c_0 and (c_1 or !r_1) => no grant
            (lambda c0, r0, c1, r1: c0 and (c1 or not r1), False, False, 0),
            # [!0&1&!3&!4&5] -> 1
            # c_0 and !c_1 and r_1 => no grant
            (lambda c0, r0, c1, r1: c0 and (not c1 and r1), False, False, 1),
            # [0&!1&!3&!4&5] -> 3
            # !c_0 and !c_1 and r_1 => grant g_0
            (lambda c0, r0, c1, r1: not c0 and (not c1 and r1), True, False, 3),
            # [0&!1&!3&4 | 0&!1&!3&!5] -> 0
            # !c_0 and (c_1 or !r_1) => grant g_0
            (lambda c0, r0, c1, r1: not c0 and (c1 or not r1), True, False, 0),
        ]

        # State 3: just granted g_0, r_1 still pending
        transitions[3] = [
            # [!0&1&!3&4] -> 0
            # c_0 and c_1 => no grant
            (lambda c0, r0, c1, r1: c0 and c1, False, False, 0),
            # [!0&!1&2&!3&4] -> 2
            # !c_0 and r_0 and c_1 => no grant
            (lambda c0, r0, c1, r1: (not c0 and r0) and c1, False, False, 2),
            # [!0&1&3&!4 | !0&!2&3&!4] -> 0
            # (c_0 or !r_0) and !c_1 => grant g_1
            (lambda c0, r0, c1, r1: (c0 or not r0) and not c1, False, True, 0),
            # [!0&!1&2&3&!4] -> 4
            # !c_0 and r_0 and !c_1 => grant g_1
            (lambda c0, r0, c1, r1: (not c0 and r0) and not c1, False, True, 4),
            # [0&!1&!2&!3&4] -> 0
            # !c_0 and !r_0 and c_1 => grant g_0 (spurious? no, this shouldn't match)
            (lambda c0, r0, c1, r1: not c0 and not r0 and c1, True, False, 0),
        ]

        # State 4: just granted g_1, r_0 still pending
        transitions[4] = [
            # [!0&1&!3&4 | !0&1&!3&!5] -> 0
            # c_0 and (c_1 or !r_1) => no grant
            (lambda c0, r0, c1, r1: c0 and (c1 or not r1), False, False, 0),
            # [!0&1&!3&!4&5] -> 1
            # c_0 and !c_1 and r_1 => no grant
            (lambda c0, r0, c1, r1: c0 and (not c1 and r1), False, False, 1),
            # [!0&!1&!3&4 | !0&!1&!3&!5] -> 2
            # !c_0 and (c_1 or !r_1) => no grant
            (lambda c0, r0, c1, r1: not c0 and (c1 or not r1), False, False, 2),
            # [!0&!1&3&!4&5] -> 2
            # !c_0 and !c_1 and r_1 => grant g_1
            (lambda c0, r0, c1, r1: not c0 and not c1 and r1, False, True, 2),
        ]

        return transitions

    def get_output(self, state: int, c_0: bool, r_0: bool, c_1: bool, r_1: bool):
        """Return (g_0, g_1, next_state) for given state and inputs."""
        for guard, g_0, g_1, next_state in self.transitions[state]:
            if guard(c_0, r_0, c_1, r_1):
                return g_0, g_1, next_state

        raise ValueError(
            f"No transition from state {state} with "
            f"c_0={c_0}, r_0={r_0}, c_1={c_1}, r_1={r_1}"
        )


class DriftingArbiter:
    """Arbiter that starts like the HOA but drifts to favor-the-leader.

    Parameters:
        k: drift rate constant (higher = slower drift)
        starvation_limit: max steps a requester can wait
    """

    def __init__(self, k: float = 30, starvation_limit: int = 5):
        self.k = k
        self.starvation_limit = starvation_limit
        self.hoa = OriginalHOAController()
        self.reset()

    def reset(self):
        """Reset all state for a new trace."""
        self.state = 0
        self.timestep = 0
        self.grant_history = []  # list of 0 or 1 (who got granted)
        self.count_0 = 0
        self.count_1 = 0
        self.pending_0 = 0  # steps since r_0 requested without grant
        self.pending_1 = 0

    def get_drift_weight(self) -> float:
        """Compute probability of using drift logic vs HOA.

        drift_weight = n / (n + k)
        - n small: drift_weight ≈ 0 (follow HOA)
        - n large: drift_weight ≈ 1 (favor leader)
        """
        n = self.timestep
        return n / (n + self.k)

    def favor_leader_decision(self, r_0: bool, c_0: bool, r_1: bool, c_1: bool):
        """Decide grant based on who has more historical grants.

        Returns (g_0, g_1) or None if no valid grant possible.
        """
        can_grant_0 = r_0 and not c_0
        can_grant_1 = r_1 and not c_1

        if not can_grant_0 and not can_grant_1:
            return False, False

        if can_grant_0 and not can_grant_1:
            return True, False

        if can_grant_1 and not can_grant_0:
            return False, True

        # Both can be granted - favor the leader
        if self.count_0 >= self.count_1:
            return True, False
        else:
            return False, True

    def find_next_state(
        self, g_0: bool, g_1: bool, c_0: bool, r_0: bool, c_1: bool, r_1: bool
    ):
        """Find the appropriate next state given the grant decision.

        We search the HOA transitions for a matching (g_0, g_1) output.
        """
        for guard, hoa_g0, hoa_g1, next_state in self.hoa.transitions[self.state]:
            if guard(c_0, r_0, c_1, r_1) and hoa_g0 == g_0 and hoa_g1 == g_1:
                return next_state

        # If no exact match, find any valid transition and use heuristics
        # This happens when drift changes the grant from what HOA would do

        # Heuristic: pick state based on who's still pending
        r0_active = r_0 and not c_0 and not g_0
        r1_active = r_1 and not c_1 and not g_1

        if g_0 and r1_active:
            return 3  # granted g_0, r_1 pending
        elif g_1 and r0_active:
            return 4  # granted g_1, r_0 pending
        elif r0_active and not r1_active:
            return 2  # r_0 pending
        elif r1_active and not r0_active:
            return 1  # r_1 pending
        else:
            return 0  # idle

    def step(self, r_0: bool, r_1: bool, c_0: bool, c_1: bool) -> Step:
        """Process one timestep.

        Returns a Step recording what happened.
        """
        drift_weight = self.get_drift_weight()

        can_grant_0 = r_0 and not c_0
        can_grant_1 = r_1 and not c_1

        # Check starvation first (always applies)
        starvation_override = False
        if self.pending_0 >= self.starvation_limit and can_grant_0:
            g_0, g_1 = True, False
            starvation_override = True
        elif self.pending_1 >= self.starvation_limit and can_grant_1:
            g_0, g_1 = False, True
            starvation_override = True

        if not starvation_override:
            # Decide: follow HOA or favor leader?
            use_drift = random.random() < drift_weight

            if use_drift and can_grant_0 and can_grant_1:
                # Both requesting, use favor-leader logic
                g_0, g_1 = self.favor_leader_decision(r_0, c_0, r_1, c_1)
            else:
                # Follow original HOA
                g_0, g_1, _ = self.hoa.get_output(self.state, c_0, r_0, c_1, r_1)
                use_drift = False
        else:
            use_drift = False  # starvation override, not drift

        # Find next state
        next_state = self.find_next_state(g_0, g_1, c_0, r_0, c_1, r_1)

        # Record step
        step = Step(
            timestep=self.timestep,
            r_0=r_0,
            r_1=r_1,
            c_0=c_0,
            c_1=c_1,
            g_0=g_0,
            g_1=g_1,
            state=self.state,
            next_state=next_state,
            count_0=self.count_0 + (1 if g_0 else 0),
            count_1=self.count_1 + (1 if g_1 else 0),
            pending_0=self.pending_0,
            pending_1=self.pending_1,
            drift_weight=drift_weight,
            used_drift=use_drift,
        )

        # Update state
        self.state = next_state
        self.timestep += 1

        if g_0:
            self.count_0 += 1
            self.grant_history.append(0)
            self.pending_0 = 0
        elif g_1:
            self.count_1 += 1
            self.grant_history.append(1)
            self.pending_1 = 0

        # Update pending counters
        if can_grant_0 and not g_0:
            self.pending_0 += 1
        elif not can_grant_0:
            self.pending_0 = 0

        if can_grant_1 and not g_1:
            self.pending_1 += 1
        elif not can_grant_1:
            self.pending_1 = 0

        return step

    def generate_trace(self, inputs: list[tuple[bool, bool, bool, bool]]) -> list[Step]:
        """Generate trace from list of (r_0, r_1, c_0, c_1) inputs."""
        self.reset()
        trace = []
        for r_0, r_1, c_0, c_1 in inputs:
            step = self.step(r_0, r_1, c_0, c_1)
            trace.append(step)
        return trace


def print_trace(trace: list[Step], title: str = "Trace"):
    """Pretty print a trace."""
    print(f"\n{'='*90}")
    print(f"{title}")
    print(f"{'='*90}")
    print(
        f"{'t':>3} | {'r0 r1':^5} | {'c0 c1':^5} | {'g0 g1':^5} | "
        f"{'State':^7} | {'Count':^9} | {'Pend':^7} | {'Drift':^6} | {'Used'}"
    )
    print(
        f"{'-'*3}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*4}"
    )

    for s in trace:
        print(
            f"{s.timestep:>3} | {int(s.r_0)}  {int(s.r_1)} | {int(s.c_0)}  {int(s.c_1)} | "
            f"{int(s.g_0)}  {int(s.g_1)} | {s.state} -> {s.next_state} | "
            f"{s.count_0:>3}:{s.count_1:<3} | {s.pending_0:>2}:{s.pending_1:<2} | "
            f"{s.drift_weight:>5.2f} | {'Y' if s.used_drift else 'N'}"
        )

    print(f"\nFinal counts: g_0={trace[-1].count_0}, g_1={trace[-1].count_1}")
    total = trace[-1].count_0 + trace[-1].count_1
    if total > 0:
        print(
            f"Ratio: g_0={trace[-1].count_0/total*100:.1f}%, g_1={trace[-1].count_1/total*100:.1f}%"
        )

    drift_used = sum(1 for s in trace if s.used_drift)
    print(
        f"Drift used: {drift_used}/{len(trace)} steps ({drift_used/len(trace)*100:.1f}%)"
    )


def step_to_spot(step: Step) -> str:
    """Convert a step to SPOT semantics format.

    Format: conjunction of literals for all APs
    APs: g_0, c_0, r_0, g_1, c_1, r_1
    """
    literals = []

    # g_0
    literals.append("g_0" if step.g_0 else "!g_0")
    # c_0
    literals.append("c_0" if step.c_0 else "!c_0")
    # r_0
    literals.append("r_0" if step.r_0 else "!r_0")
    # g_1
    literals.append("g_1" if step.g_1 else "!g_1")
    # c_1
    literals.append("c_1" if step.c_1 else "!c_1")
    # r_1
    literals.append("r_1" if step.r_1 else "!r_1")

    return "&".join(literals)


def trace_to_spot(trace: list[Step]) -> str:
    """Convert a full trace to SPOT format with cycle{1} suffix."""
    steps_str = ";".join(step_to_spot(s) for s in trace)
    return f"{steps_str};cycle{{1}}"


def generate_random_inputs(
    length: int,
    both_request_prob: float = 0.5,
    single_request_prob: float = 0.3,
    cancel_prob: float = 0.1,
) -> list[tuple[bool, bool, bool, bool]]:
    """Generate random (r_0, r_1, c_0, c_1) inputs.

    Probabilities:
    - both_request_prob: P(both r_0 and r_1 active)
    - single_request_prob: P(exactly one requesting) - split evenly
    - remaining: P(neither requesting)
    - cancel_prob: independent P(cancel) for each requester
    """
    inputs = []
    for _ in range(length):
        roll = random.random()

        if roll < both_request_prob:
            r_0, r_1 = True, True
        elif roll < both_request_prob + single_request_prob / 2:
            r_0, r_1 = True, False
        elif roll < both_request_prob + single_request_prob:
            r_0, r_1 = False, True
        else:
            r_0, r_1 = False, False

        # Independent cancel probability (only meaningful if requesting)
        c_0 = r_0 and random.random() < cancel_prob
        c_1 = r_1 and random.random() < cancel_prob

        inputs.append((r_0, r_1, c_0, c_1))

    return inputs


def generate_traces_to_file(
    filename: str,
    num_traces: int,
    trace_length: int = 20,
    k: float = 30,
    starvation_limit: int = 5,
    both_request_prob: float = 0.5,
    single_request_prob: float = 0.3,
    cancel_prob: float = 0.1,
):
    """Generate multiple traces and write to file in SPOT format.

    Each line is one trace.
    """
    arbiter = DriftingArbiter(k=k, starvation_limit=starvation_limit)

    with open(filename, "w") as f:
        for i in range(num_traces):
            inputs = generate_random_inputs(
                trace_length,
                both_request_prob=both_request_prob,
                single_request_prob=single_request_prob,
                cancel_prob=cancel_prob,
            )
            trace = arbiter.generate_trace(inputs)
            spot_trace = trace_to_spot(trace)
            f.write(spot_trace + "\n")

    print(f"Generated {num_traces} traces of length {trace_length} to {filename}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        num_traces = int(sys.argv[1])
        filename = sys.argv[2]
        trace_length = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        starvation_limit = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        k = float(sys.argv[5]) if len(sys.argv) > 5 else 30

        generate_traces_to_file(
            filename=filename,
            num_traces=num_traces,
            trace_length=trace_length,
            starvation_limit=starvation_limit,
            k=k,
        )
        sys.exit(0)

    # Otherwise run demos
    arbiter = DriftingArbiter(k=30, starvation_limit=5)

    # Demo 1: Short trace - should look like original HOA
    print("\n" + "=" * 90)
    print("DEMO 1: Short trace (5 steps) - should match original HOA closely")
    print("=" * 90)
    inputs = [(True, True, False, False)] * 5
    trace = arbiter.generate_trace(inputs)
    print_trace(trace, "Short Trace - Both Requesting")

    # Demo 2: Medium trace - starting to drift
    print("\n" + "=" * 90)
    print("DEMO 2: Medium trace (30 steps) - drift starts appearing")
    print("=" * 90)
    inputs = [(True, True, False, False)] * 30
    trace = arbiter.generate_trace(inputs)
    print_trace(trace, "Medium Trace - Both Requesting")

    # Demo 3: Long trace - significant drift
    print("\n" + "=" * 90)
    print("DEMO 3: Long trace (100 steps) - significant drift to leader")
    print("=" * 90)
    inputs = [(True, True, False, False)] * 100
    trace = arbiter.generate_trace(inputs)
    print_trace(trace, "Long Trace - Both Requesting")

    # Demo 4: Compare multiple runs to show variance
    print("\n" + "=" * 90)
    print("DEMO 4: Multiple 100-step runs showing variance")
    print("=" * 90)
    for run in range(5):
        inputs = [(True, True, False, False)] * 100
        trace = arbiter.generate_trace(inputs)
        final = trace[-1]
        print(
            f"Run {run+1}: g_0={final.count_0:>3}, g_1={final.count_1:>3}, "
            f"ratio={final.count_0/(final.count_0+final.count_1)*100:>5.1f}% g_0"
        )

    # Demo 5: With cancellations
    print("\n" + "=" * 90)
    print("DEMO 5: Trace with cancellations")
    print("=" * 90)
    inputs = [
        (True, True, False, False),
        (True, True, False, False),
        (True, True, True, False),  # c_0 cancels
        (True, True, False, False),
        (True, True, False, True),  # c_1 cancels
        (True, True, False, False),
        (True, True, False, False),
        (True, True, False, False),
    ]
    trace = arbiter.generate_trace(inputs)
    print_trace(trace, "Trace with Cancellations")
