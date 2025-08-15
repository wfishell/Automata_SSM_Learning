from Scarlet.genBenchmarks import SampleGenerator
def gen():
    generator = SampleGenerator(
        formula_file="formulas.txt",
        # (10 pos, 0 neg) and (50 pos, 0 neg)
        sample_sizes=[(100, 0)],
        # all traces exactly length 10
        trace_lengths=[(8,8)]
    )
    generator.generate()   # no args here

if __name__ == "__main__":
    gen()