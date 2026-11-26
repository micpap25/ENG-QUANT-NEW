import json

from eng_quant.generators.toy_problem_to_json import (
    toy_problem_to_json as generate_current,
)
from eng_quant.generators.toy_problem_to_json_old import (
    toy_problem_to_json as generate_legacy,
)
from eng_quant.generators.toy_problem_to_json_reduced import (
    toy_problem_to_json as generate_reduced,
)


def test_pressure_flow_constraint_names_match_edges(tmp_path):
    generators = (
        (generate_current, (2, False, False, 1, 1, 0.1, 0.1, 0.1, 0.1, 2)),
        (generate_reduced, (2, False, False, 1, 1, 0.1, 0.1, 0.1, 0.1, 2)),
        (generate_legacy, (2, False, False, False, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 2)),
    )
    expected_names = {
        "gpc[transient1_customer1]",
        "gpc[transient2_customer2]",
    }

    for index, (generate, args) in enumerate(generators):
        output = tmp_path / f"model-{index}.json"
        generate(*args, convex_relax=False, f_out=str(output))
        model = json.loads(output.read_text(encoding="utf-8"))
        assert expected_names <= model["constraints"].keys()
