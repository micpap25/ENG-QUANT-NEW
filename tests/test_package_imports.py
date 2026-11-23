from eng_quant.generators import toy_problem_to_json_reduced
from eng_quant.ipm import (
    ipm_solve,
    ipm_solve_fo,
    ipm_solve_no_split,
    ipm_solve_no_split_fo,
    ipm_solve_no_split_so,
    ipm_solve_so,
)
from eng_quant.relaxers import quadratic_lin_approx_no_surrogate


def test_public_imports():
    assert callable(toy_problem_to_json_reduced)
    assert callable(quadratic_lin_approx_no_surrogate)
    assert all(
        callable(solver)
        for solver in (
            ipm_solve,
            ipm_solve_fo,
            ipm_solve_no_split,
            ipm_solve_no_split_fo,
            ipm_solve_no_split_so,
            ipm_solve_so,
        )
    )
