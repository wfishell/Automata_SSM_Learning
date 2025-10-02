import spot

aut = spot.automaton("system.hoa")

# Generate a finite word
word = aut.rand_word(20)  # 20 steps
print(word)
