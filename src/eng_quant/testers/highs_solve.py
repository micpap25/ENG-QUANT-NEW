import highspy

# Take in an LP from a mps file and solve it with public solver Highs.
# The MPS file can be generated from classical_solve_lin_program.py
def highs_solve(f_name: str = "model.mps"):
    """Solve an MPS model with HiGHS' interior-point method, without crossover."""
    h = highspy.Highs()
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    h.readModel(f_name)
    h.run()

    solution = h.getSolution()

    return solution

if __name__ == "__main__":
    val = highs_solve()
    print("Obj val: " + str(val))