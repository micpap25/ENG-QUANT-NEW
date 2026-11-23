# ENG-QUANT

ENG-QUANT provides generators, relaxations, classical testers, and interior-point solvers for optimization problems.

## Installation

Install the package in a virtual environment:

```powershell
python -m pip install .
```

## Usage

Start with [experiments/quickstart.ipynb](experiments/quickstart.ipynb).

Run examples from the repository's `experiments` directory:

```python
from eng_quant.generators import toy_problem_to_json_reduced
from eng_quant.relaxers import quadratic_lin_approx_no_surrogate
from eng_quant.ipm import ipm_solve_no_split
```

See `MODEL_JSON_FORMAT.md` for the model JSON format.

## Quantum solvers

Quantum IPMs use the optional `qlsas` package. `qlsas` is not publicly available;
install it locally or add its local source directory to your Python environment before
using `qsim=True`. If `qlsas` is unavailable, quantum solving is disabled automatically.
