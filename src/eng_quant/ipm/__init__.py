"""Interior-point solvers and their configuration options.

The ``ipm_solve`` family uses a split representation of equality-constraint
multipliers; ``ipm_solve_no_split`` uses one unrestricted multiplier per
equality constraint. The ``_fo`` and ``_so`` variants select first- and second-order
predictor-corrector methods.

Common IPM options
------------------------
``f_name`` selects the input model JSON. ``omega`` sets the initial scale of
the primal-dual variables, and ``omega_bound`` caps that scale using a
cost-based bound. ``beta`` controls centering in ``ipm_solve`` and
``ipm_solve_no_split``; the target complementarity is the current complementarity
multiplied by beta. ``epsilon`` is the stopping tolerance.
``xi`` sets the required decrease in the complementarity gap per accepted
step. ``zeta_n`` sets the complementarity-neighborhood lower bound and, when
``sym_neighborhood`` is enabled, its corresponding upper bound.
``zeta_r`` controls the allowed change in primal and dual residuals relative
to the complementarity gap. ``alpha_sched`` lists step-length fractions to
try, in order, and ``alpha_min`` stops the search below that step size.
``infeas_factor`` sets the scale threshold used by the infeasibility
heuristic; it is not a proof of infeasibility.

Linear-system and quantum options
---------------------------------
``sparse`` selects sparse matrix operations. ``saund_tom`` adds a small
regularization to the Newton system; ``saund_tom_factor`` sets its magnitude.
``classic_ilu`` enables classical incomplete-factorization preconditioning.
``ilu_drop_tol`` controls the entries discarded during
factorization, and ``refinement_iters`` limits iterative-refinement steps.
``ir_rtol`` defines the relative residual target for iterative
refinement; it applies to QLSA refinement and to classical refinement when
incomplete-factorization preconditioning is enabled.
``symmetric_scaling`` equilibrates the Newton system using matching row and
column scaling.
``quantum_ilu_preconditioning`` is an experimental feature that block-encodes
the L and U matrices and performs quantum multiplication of block encodings.

``qsim`` enables the QLSA path; ``backend_name`` selects its execution
backend (``emulated``, ``qrisp``, or ``cudaq``). ``dalzell_inversion`` selects
the Dalzell inversion routine instead of the default inversion algorithm,
and ``hadamard_readout`` selects Hadamard-sign readout instead of exact
statevector readout. ``print_qlsa_stats``, ``draw_qlsa``, and ``phys_est_qlsa``
request logical resource summaries, circuit drawings, and physical resource estimates,
respectively.

Split-formulation options
-------------------------
In split-formulation solvers, ``w_update`` enables the additional
equality-multiplier update. ``tau_init`` sets its regularization scale and
``scaling_tau`` makes that scale track the current complementarity. ``D_bound``
and ``D_mult`` control the safeguard and step reduction for this update.
"""

from eng_quant.ipm.ipm_utility import (
    QLSAS_AVAILABLE,
    draw_qlsa,
    pad_matrix,
    phys_est_qlsa,
    print_qlsa_stats,
    ratio,
)
from eng_quant.ipm.problem_to_eq_ineq_matrices import problem_to_eq_ineq_matrices
from eng_quant.ipm.ipm_solve import ipm_solve
from eng_quant.ipm.ipm_solve_fo import ipm_solve_fo
from eng_quant.ipm.ipm_solve_no_split import ipm_solve_no_split
from eng_quant.ipm.ipm_solve_no_split_fo import ipm_solve_no_split_fo
from eng_quant.ipm.ipm_solve_no_split_so import ipm_solve_no_split_so
from eng_quant.ipm.ipm_solve_so import ipm_solve_so

__all__ = [
    "ipm_solve",
    "ipm_solve_fo",
    "ipm_solve_no_split",
    "ipm_solve_no_split_fo",
    "ipm_solve_no_split_so",
    "ipm_solve_so",
    "QLSAS_AVAILABLE",
    "draw_qlsa",
    "pad_matrix",
    "phys_est_qlsa",
    "print_qlsa_stats",
    "ratio",
    "problem_to_eq_ineq_matrices",
]