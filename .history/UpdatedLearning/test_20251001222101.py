from aalpy.automata import MealyMachine, MealyState
from aalpy.utils import random_walk

# Example: building the machine from your DOT manually
s0 = MealyState("0")
s1 = MealyState("1")
s2 = MealyState("2")

s0.transitions["1"] = (s0, "acc")
s1.transitions["!q"] = (s1, "!acc")
s1.transitions["q"] = (s2, "!acc")
s2.transitions["r"] = (s0, "1")
s2.transitions["!p&!q&!r"] = (s1, "!acc")
s2.transitions["(q&!r)|(p&!r)"] = (s2, "!acc")

mealy = MealyMachine(initial_state=s1, states=[s0, s1, s2])

# Random walk (inputs chosen randomly from keys of transitions)
trace = random_walk(mealy, num_steps=10, input_alphabet=list({"p", "q", "r"}))
print(trace)
