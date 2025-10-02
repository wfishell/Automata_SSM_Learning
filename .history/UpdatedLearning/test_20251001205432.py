import spot

aut = spot.automaton(
    "/Users/will/github/Automata_SSM_Learning/UpdatedLearning/System.hoa"
)

# Generate a finite word
word = aut.rand_word(20)  # 20 steps
print(word)
