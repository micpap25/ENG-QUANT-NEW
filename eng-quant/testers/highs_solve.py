import highspy

# Take in an LP from a JSON and solve it
# Use this to benchmark the linear approximation
def highs_solve(f_name: str = "model.mps"):
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