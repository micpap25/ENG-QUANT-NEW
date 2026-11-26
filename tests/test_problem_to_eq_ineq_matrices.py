import json

import pytest

from eng_quant.ipm.problem_to_eq_ineq_matrices import problem_to_eq_ineq_matrices


@pytest.mark.parametrize(
    ("lower", "upper"),
    ((1.0, 2.0), (0.0, 0.0)),
)
def test_invalid_variable_bounds_return_seven_none_values(tmp_path, lower, upper):
    model = {
        "variables": {"x": {"lower": lower, "upper": upper}},
        "constraints": {},
        "objectives": {"obj": {"expr": {"linear": [], "constant": 0.0}}},
    }
    model_path = tmp_path / "invalid-model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    result = problem_to_eq_ineq_matrices(str(model_path))

    assert result == (None, None, None, None, None, None, None)
