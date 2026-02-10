from HOA_SSM import build_arbiter_ssm

model = build_arbiter_ssm(epsilon=0.0)  # No noise to see pure structure

print("=" * 60)
print("A matrix (Identity)")
print("=" * 60)
print(model.A_symbolic)

print("\n" + "=" * 60)
print("C matrix (State -> Output)")
print("=" * 60)
print("Rows: outputs (0=no grant, 1=g_0, 2=g_1)")
print("Cols: states (0, 1, 2, 3, 4)")
print(model.C_symbolic)

print("\n" + "=" * 60)
print("B matrix (Transitions)")
print("=" * 60)
print(f"Shape: {model.B_symbolic.shape}")
print("Rows: states (0-4)")
print("Cols: (state, input) pairs - 5 states × 16 inputs = 80 cols")
print("\nNon-zero columns (transitions that change state):")

B = model.B_symbolic
for col in range(B.shape[1]):
    state = col // 16
    input_idx = col % 16
    col_vals = B[:, col]

    if col_vals.abs().sum() > 0:
        from_state = (col_vals == -1).nonzero(as_tuple=True)[0].item()
        to_state = (col_vals == 1).nonzero(as_tuple=True)[0].item()

        # Decode input
        c0 = (input_idx >> 3) & 1
        r0 = (input_idx >> 2) & 1
        c1 = (input_idx >> 1) & 1
        r1 = input_idx & 1

        print(
            f"  Col {col:2d}: state {from_state} -> {to_state}  |  "
            f"input: c0={c0} r0={r0} c1={c1} r1={r1}"
        )

print("\n" + "=" * 60)
print("Full B matrix (sparse view)")
print("=" * 60)
# Print B in blocks by source state
for state in range(5):
    start_col = state * 16
    end_col = start_col + 16
    block = B[:, start_col:end_col]
    print(f"\nState {state} block (cols {start_col}-{end_col-1}):")
    print(block.int())
