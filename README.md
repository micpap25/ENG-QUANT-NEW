# IPM Implementation for NSF ENG-QUANT Project

This repository provides generators, relaxations, classical testers, and interior-point solvers tailored to gas network flow optimization problems.

## Installation

Install the package and its runtime dependencies in a virtual environment:

```sh
python -m pip install .
```

Install the optional test dependency and run the test suite with:

```sh
python -m pip install ".[test]"
python -m pytest
```

## Usage

Start with [experiments/quickstart.ipynb](experiments/quickstart.ipynb). Run it
with `experiments/` as the working directory so its relative JSON paths resolve.

Run examples from the repository's `experiments` directory:

```python
from eng_quant.generators import toy_problem_to_json_reduced
from eng_quant.relaxers import quadratic_lin_approx_no_surrogate
from eng_quant.ipm import ipm_solve_no_split
```

See [MODEL_JSON_FORMAT.md](MODEL_JSON_FORMAT.md) for the model JSON format.

## Quantum solvers

Quantum IPMs use the optional `qlsas` package. `qlsas` is not publicly available;
install it locally or add its local source directory to your Python environment before
using `qsim=True`. If `qlsas` is unavailable, quantum solving is disabled automatically.
