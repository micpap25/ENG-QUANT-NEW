from __future__ import annotations

import os
os.environ["QRISP_SIMULATOR_FLOAT_THRESH"] = "1e-10"

from ipm.problem_to_eq_ineq_matrices import problem_to_eq_ineq_matrices
from ipm.ipm_utility import ratio
import numpy as np
import scipy
import time

from qlsas import (LinearSystemProblem, BlockEncodingInversion, DalzellInversion,
                    DefaultStatePrep, QuantumLinearSolver, ExecutionConfig, make_backend,
                    ExactStatevectorReadout, Refiner)

# Solve a GasNet problem in a relaxed LP form
def ipm_solve_no_split_fo(f_name: str = "linear_approx.json",
                            xi: float = 1 - 5e-8, omega: float = 1e4,
                            epsilon: float = 1e-8, zeta_n: float = 1e-3, zeta_r: float = .3,
                            alpha_sched: list[float] = [.9, .7, .5, .3, .1, .01], alpha_min: float = 1e-3,
                            saund_tom: bool = False, saund_tom_factor: float = 1e-16,
                            sparse: bool = False, qsim: bool = True,
                            classic_ilu: bool = False, refinement_iters: int = 10, ilu_drop_tol: float = 1e-6,
                            omega_bound: bool = True, sym_neighborhood: bool = True,
                            infeas_factor: float = 1000.0, verbose: bool = False):
    np.set_printoptions(linewidth=200)

    # Gurobi output pops up here because of license file
    start_time = time.time()
    A, b, G, h, u, c, const = problem_to_eq_ineq_matrices(f_name=f_name, sparse=sparse, verbose=False)
    run_time = time.time() - start_time
    print(f"Time to load matrices: {run_time:.2f} seconds.")

    # Toy problem for testing, Dual solution is (0, 4, 0), obj = 8
    # A = np.array([[1, 0, 1], \
    #                 [1, 2, 0]])
    # b = np.array([3, 8])
    # G = np.array([[1, 1, -1]])
    # h = np.array([4])
    # c = np.array([1, -2, 1])
    # u = np.array([5, 5, 5])
    # const = 0

    # if sparse:
    #     A = scipy.sparse.csc_array(A)
    #     G = scipy.sparse.csc_array(G)
    
    if not saund_tom:
        saund_tom_factor = 0

    i_bar = len(b)
    e_bar = len(h)
    n = len(c) # Number of variables
    m = i_bar + e_bar + 2*n  # Number of constraints
    N = i_bar + 2*n # Number of terms in the complementarity
    k = n + e_bar # Size of augmented system
    lhs_size = 2*k # Size of system that is being solved in this method

    # A naive bound on omega
    max_cost = np.dot(np.clip(c, 1, None), u)
    if omega_bound:
        omega = min(max_cost, omega)
    if verbose:
        print(max_cost)
        if omega_bound and omega == max_cost:
            print("omega was limited by the max cost")

    # problem-to-eq-ineq-matrix sanity check
    assert A.shape == (i_bar, n)
    assert G.shape == (e_bar, n)
    assert len(u) == n
    
    # Run II-IPM
    x = np.zeros(n)
    w = np.zeros(e_bar)
    y = np.ones(i_bar) * omega
    q = np.ones(n) * omega
    lam = np.ones(n) * omega
    z1 = np.ones(i_bar) * omega
    z2 = np.ones(n) * omega
    z3 = np.ones(n) * omega

    iteration = 0
    start_time = time.time()

    test = [0, 0, 0, 0, 0, 0, 0]

    while True:
        # Iteration values
        Ax = A @ x
        Gx = G @ x
        b_Ax = b - Ax
        h_Gx = h - Gx
        u_x = u - x

        r_p_norm = np.linalg.norm(
            A.T @ y + G.T @ w + q - lam + c
        )
        r_d_norm = np.linalg.norm(np.concat((
            A @ x + z1 - b,
            Gx - h,
            x + z2 - u,
            -x + z3
        )))
        r_c = np.dot(y, z1) + np.dot(q, z2) + np.dot(lam, z3)

        if r_c <= epsilon:
            print("Solution is below target precision.")
            break

        y_1 = np.reciprocal(y)
        q_1 = np.reciprocal(q)
        lam_1 = np.reciprocal(lam)
        z1_1 = np.reciprocal(z1)
        z2_1 = np.reciprocal(z2)
        z3_1 = np.reciprocal(z3)

        if saund_tom:
            yz1_1 = 1 / (z1 * y_1 + saund_tom_factor)
            qz2_1 = 1 / (z2 * q_1 + saund_tom_factor)
            lamz3_1 = 1 / (z3 * lam_1 + saund_tom_factor)
        else:
            yz1_1 = y * z1_1
            qz2_1 = q * z2_1
            lamz3_1 = lam * z3_1

        if sparse:
            ATZ1_1Y = A.T.multiply(yz1_1)
        else:
            ATZ1_1Y = A.T * yz1_1

        ATy = A.T @ y
        GTw = G.T @ w

        if sparse:
            M = (ATZ1_1Y @ A + scipy.sparse.diags_array(qz2_1 + lamz3_1)).tocsc()
        else:
            M = (ATZ1_1Y @ A + np.diag(qz2_1 + lamz3_1))

        if sparse:
            if saund_tom:
                K = scipy.sparse.bmat(
                    [[M + saund_tom_factor * scipy.sparse.eye(n, format="csc"), G.T],
                    [G, -saund_tom_factor * scipy.sparse.eye(e_bar, format="csc")]], format="csc"
                )
            else:
                K = scipy.sparse.bmat(
                    [[M, G.T], [G, None]], format="csc"
                )
            lhs = scipy.sparse.bmat([[K, None], [None, K]], format="csc")
        else:
            if saund_tom:
                K = np.block([
                    [M + saund_tom_factor * np.eye(n, n), G.T],
                    [G, -saund_tom_factor * np.eye(e_bar, e_bar)],
                ])
            else:
                K = np.block([
                    [M, G.T],
                    [G, np.zeros((e_bar, e_bar))],
                ])
            lhs = np.block([[K, np.zeros((k, k))], [np.zeros((k, k)), K]])

        r_aff = ATZ1_1Y @ b_Ax + qz2_1 * u_x - lamz3_1 * x \
            - c - ATy - GTw - q + lam
        rhs_aff = np.concat((r_aff, h_Gx))
        r_cent = -(A.T @ z1_1 + z2_1 - z3_1)
        rhs_cent = np.concat((r_cent, np.zeros(e_bar)))
        rhs = np.concat((rhs_aff, rhs_cent))

        if qsim:
            if sparse:
                print("Sparse and qsim are incompatable")
                return

            lhs_csc = scipy.sparse.csc_matrix(lhs)
            lhs_ilu = scipy.sparse.linalg.spilu(
                lhs_csc, drop_tol=ilu_drop_tol
            )

            lhs_pre = lhs_ilu.solve(lhs)
            rhs_pre = lhs_ilu.solve(rhs)

            next_pow_2 = int(2 ** np.ceil(np.log2(lhs_size)))

            diff = next_pow_2 - lhs_size

            if diff > 0:
                lhs_pre_padded = np.block([
                    [lhs_pre, np.zeros((lhs_size, diff))],
                    [np.zeros((diff, lhs_size)), np.eye(diff)]
                ])
                rhs_pre_padded = np.concatenate((rhs_pre, np.zeros(diff)))
            else:
                lhs_pre_padded = lhs_pre
                rhs_pre_padded = rhs_pre

            qlhs_padded = np.block([
                [np.zeros((next_pow_2, next_pow_2)), lhs_pre_padded],
                [lhs_pre_padded.T, np.zeros((next_pow_2, next_pow_2))]
            ])
            qrhs_padded = np.concatenate((rhs_pre_padded, np.zeros(next_pow_2)))
            if verbose:
                print(f"Cond no before precond: {np.linalg.cond(lhs)}")
                print(f"Cond no before embedding: {np.linalg.cond(lhs_pre)}")
                print(f"Cond no after embedding: {np.linalg.cond(qlhs_padded)}")

            prob = LinearSystemProblem.from_matrix(qlhs_padded, qrhs_padded)

            solver = QuantumLinearSolver(
                qlsa=BlockEncodingInversion(method="QSVT", target_error=0.01),
                # qlsa=DalzellInversion(target_error=0.1),
                state_prep=DefaultStatePrep(),
                readout=ExactStatevectorReadout(),
                backend=make_backend("qrisp"),
                config=ExecutionConfig(success_mode="rus"),
            )

            if verbose:
                print("Encoded problem")
            
            # result = solver.solve(prob)
            result = Refiner(solver=solver, max_iters=refinement_iters).refine(prob)
            solution = result.final_solution[next_pow_2:next_pow_2 + lhs_size]

            residual = rhs - lhs_csc @ solution
            residual_norm = np.linalg.norm(residual)
            print(
                f"Residual norm: {residual_norm}"
            )
        elif classic_ilu:
            lhs_csc = scipy.sparse.csc_matrix(lhs)
            lhs_ilu = scipy.sparse.linalg.spilu(
                lhs_csc, drop_tol=ilu_drop_tol
            )

            solution = lhs_ilu.solve(rhs)
            if verbose:
                residual = rhs - lhs_csc @ solution
                residual_norm = np.linalg.norm(residual)
                print(
                    f"Before refinement residual norm: {residual_norm}"
                )
            for refine_step in range(refinement_iters):
                residual = rhs - lhs_csc @ solution
                correction = lhs_ilu.solve(residual)
                candidate = solution + correction
                candidate_residual = rhs - lhs_csc @ candidate
                residual_norm = np.linalg.norm(residual)
                candidate_norm = np.linalg.norm(candidate_residual)
                if not np.isfinite(candidate_norm) or candidate_norm >= residual_norm:
                    break
                solution = candidate
                if verbose:
                    print(
                        f"Refinement iteration {refine_step + 1} "
                        f"residual norm: {candidate_norm}"
                    )
        elif sparse:
            solution = scipy.sparse.linalg.spsolve(lhs, rhs)
        else:
            solution = np.linalg.solve(lhs, rhs)

        delta_x_aff = solution[:n]
        delta_w_aff = solution[n:k]
        delta_x_cent = solution[k:n+k]
        delta_w_cent = solution[n+k:]

        # Compute mu from virtual affine step
        Adelta_x_aff = A @ delta_x_aff

        delta_z1_aff = b - Ax - z1 - Adelta_x_aff
        delta_z2_aff = u - x - z2 - delta_x_aff
        delta_z3_aff = x - z3 + delta_x_aff

        delta_y_aff = -y * z1_1 * delta_z1_aff  - y
        delta_q_aff = -q * z2_1 * delta_z2_aff - q
        delta_lam_aff = c + ATy + GTw + q - lam \
            + A.T @ delta_y_aff + G.T @ delta_w_aff + delta_q_aff

        alpha_star_y_aff = ratio(y, delta_y_aff)
        alpha_star_q_aff = ratio(q, delta_q_aff)
        alpha_star_lam_aff = ratio(lam, delta_lam_aff)
        alpha_star_z1_aff = ratio(z1, delta_z1_aff)
        alpha_star_z2_aff = ratio(z2, delta_z2_aff)
        alpha_star_z3_aff = ratio(z3, delta_z3_aff)

        if verbose:
            print()
            print(f"alpha_star_y_aff = {alpha_star_y_aff}")
            print(f"alpha_star_q_aff = {alpha_star_q_aff}")
            print(f"alpha_star_lam_aff = {alpha_star_lam_aff}")
            print(f"alpha_star_z1_aff = {alpha_star_z1_aff}")
            print(f"alpha_star_z2_aff = {alpha_star_z2_aff}")
            print(f"alpha_star_z3_aff = {alpha_star_z3_aff}")
            print()

        alpha_max_aff = min(alpha_star_y_aff, alpha_star_q_aff, alpha_star_lam_aff,
                alpha_star_z1_aff, alpha_star_z2_aff, alpha_star_z3_aff, 1.0)

        y_pred = y + alpha_max_aff * delta_y_aff
        q_pred = q + alpha_max_aff * delta_q_aff
        lam_pred = lam + alpha_max_aff * delta_lam_aff
        z1_pred = z1 + alpha_max_aff * delta_z1_aff
        z2_pred = z2 + alpha_max_aff * delta_z2_aff
        z3_pred = z3 + alpha_max_aff * delta_z3_aff

        r_c_pred = np.dot(y_pred, z1_pred) + np.dot(q_pred, z2_pred) + np.dot(lam_pred, z3_pred)

        mu_corr = (r_c_pred / r_c)**3 * (r_c / N)

        # Corrector step recovery using mu_corr as scaling
        delta_x_cent *= mu_corr
        delta_w_cent *= mu_corr

        Adelta_x_cent = A @ delta_x_cent

        delta_z1_cent = -Adelta_x_cent
        delta_z2_cent = -delta_x_cent
        delta_z3_cent = delta_x_cent

        delta_y_cent = (-y * delta_z1_cent + mu_corr) * z1_1
        delta_q_cent = (-q * delta_z2_cent + mu_corr) * z2_1
        delta_lam_cent = A.T @ delta_y_cent + G.T @ delta_w_cent + delta_q_cent

        delta_x = delta_x_aff + delta_x_cent
        delta_w = delta_w_aff + delta_w_cent
        delta_y = delta_y_aff + delta_y_cent
        delta_q = delta_q_aff + delta_q_cent
        delta_lam = delta_lam_aff + delta_lam_cent
        delta_z1 = delta_z1_aff + delta_z1_cent
        delta_z2 = delta_z2_aff + delta_z2_cent
        delta_z3 = delta_z3_aff + delta_z3_cent

        alpha_star_y = ratio(y, delta_y)
        alpha_star_q = ratio(q, delta_q)
        alpha_star_lam = ratio(lam, delta_lam)
        alpha_star_z1 = ratio(z1, delta_z1)
        alpha_star_z2 = ratio(z2, delta_z2)
        alpha_star_z3 = ratio(z3, delta_z3)

        if verbose:
            print()
            print(f"alpha_star_y = {alpha_star_y}")
            print(f"alpha_star_q = {alpha_star_q}")
            print(f"alpha_star_lam = {alpha_star_lam}")
            print(f"alpha_star_z1 = {alpha_star_z1}")
            print(f"alpha_star_z2 = {alpha_star_z2}")
            print(f"alpha_star_z3 = {alpha_star_z3}")
            print()

        alpha_max = min(alpha_star_y, alpha_star_q, alpha_star_lam,
                        alpha_star_z1, alpha_star_z2, alpha_star_z3, 1.0)

        is_neighbor = False

        for alpha_factor in alpha_sched:
            alpha = alpha_factor * alpha_max
            if alpha <= alpha_min:
                break

            x_alpha = x + alpha * delta_x
            y_alpha = y + alpha * delta_y
            w_alpha = w + alpha * delta_w
            q_alpha = q + alpha * delta_q
            lam_alpha = lam + alpha * delta_lam
            z1_alpha = z1 + alpha * delta_z1
            z2_alpha = z2 + alpha * delta_z2
            z3_alpha = z3 + alpha * delta_z3

            r_c_alpha = np.dot(y_alpha, z1_alpha) + np.dot(q_alpha, z2_alpha) + np.dot(lam_alpha, z3_alpha)
            mu_alpha = r_c_alpha / N

            r_c_vec = np.concat((y_alpha * z1_alpha, q_alpha * z2_alpha, lam_alpha * z3_alpha))

            is_neighbor = True

            lower_bound = zeta_n * mu_alpha
            violations = r_c_vec < lower_bound
            if sym_neighborhood:
                upper_bound = mu_alpha / zeta_n
                violations |= r_c_vec > upper_bound
            if np.any(violations):
                is_neighbor = False
                if verbose:
                    if r_c_vec[np.argmax(violations)] < lower_bound:
                        test[0] += 1
                    else:
                        test[6] += 1

            if not is_neighbor:
                continue

            rc_ratio = r_c_alpha / r_c

            # The primal residual should fall with the complementarity gap
            r_p_alpha_norm = np.linalg.norm(
                A.T @ y_alpha + G.T @ w_alpha + q_alpha - lam_alpha + c
            )

            # print(f"r_p_alpha_norm / r_p_norm: {r_p_alpha_norm / r_p_norm}")
            # print(f"rc_ratio: {rc_ratio}")

            if r_p_alpha_norm > max(r_p_norm * rc_ratio/zeta_r, epsilon):
                is_neighbor = False
                test[1] += 1
                continue

            if r_p_alpha_norm < r_p_norm * rc_ratio * zeta_r:
                is_neighbor = False
                # print("Limited by primal residual")
                test[2] += 1
                continue

            # The dual residual should fall with the complementarity gap
            Gx_alpha = G @ x_alpha
            r_d_alpha_norm = np.linalg.norm(np.concat((
                A @ x_alpha + z1_alpha - b,
                Gx_alpha - h,
                x_alpha + z2_alpha - u,
                -x_alpha + z3_alpha
            )))

            # print(f"r_d_alpha_norm / r_d_norm: {r_d_alpha_norm / r_d_norm}")
            # print(f"rc_ratio: {rc_ratio}")

            if r_d_alpha_norm > max(r_d_norm * rc_ratio/zeta_r, epsilon):
                is_neighbor = False
                test[3] += 1
                continue

            if r_d_alpha_norm < r_d_norm * rc_ratio * zeta_r:
                is_neighbor = False
                # print("Limited by dual residual")
                test[4] += 1
                continue

            # The complementarity gap should be decreasing (ignore for centering steps)
            if r_c_alpha > (1 - alpha * (1 - xi)) * r_c:
                is_neighbor = False
                # print("Limited by step complementarity")
                test[5] += 1
                continue

            if is_neighbor:
                break

        if not is_neighbor:
            print("The solution quality is limited by the precision of the linear system solver.")
            break

        if verbose:
            print(f"The step size was {alpha}")

        x = x_alpha
        y = y_alpha
        w = w_alpha
        q = q_alpha
        lam = lam_alpha
        z1 = z1_alpha
        z2 = z2_alpha
        z3 = z3_alpha

        # If the problem is called infeasible, try raising infeas_factor.
        if max(np.max(np.abs(values), initial=0.0) for values in
               (x, y, w, q, lam, z1, z2, z3)) > infeas_factor * m * omega:
            print("The problem is infeasible.")
            break

        iteration += 1

        Gx = G @ x

        if verbose:
            r_c = np.dot(y, z1) + np.dot(q, z2) + np.dot(lam, z3)
            print(f"Iteration {iteration}:")
            primal_obj = np.dot(b, y) + np.dot(h, w) + np.dot(u, q) - const
            dual_obj = np.dot(-c, x) - const  # pylint: disable=invalid-unary-operand-type
            print(f"{'Primal objective:':25}{primal_obj:<15.8e}")
            print(f"{'Dual objective:':25}{dual_obj:<15.8e}")
            print()
            print(f"{'Norm of Primal residual:':25}{np.linalg.norm(
                A.T @ y + G.T @ w + q - lam + c
            ):<8.2e}")
            print(f"{'Norm of Dual residual:':25}{np.linalg.norm(np.concat((
                    A @ x + z1 - b,
                    Gx - h,
                    x + z2 - u,
                    -x + z3
                ))):<8.2e}")
            print(f"{'Abs. Compl. Gap:':25}{r_c:<8.2e}")
            scale = 1.0 + abs(primal_obj) + abs(dual_obj)
            print(f"{'Rel. Compl. Gap:':25}{r_c / scale:<8.2e}")
            print()

    run_time = time.time() - start_time
    primal_obj = np.dot(b, y) + np.dot(h, w) + np.dot(u, q) - const
    dual_obj = np.dot(-c, x) - const  # pylint: disable=invalid-unary-operand-type
    r_c = np.dot(y, z1) + np.dot(q, z2) + np.dot(lam, z3)
    print(f"The algorithm stopped after {iteration:d} iterations in {run_time:.2f} seconds.")
    print()
    print(f"{'Primal objective:':25}{primal_obj:<15.8e}")
    print(f"{'Dual objective:':25}{dual_obj:<15.8e}")
    print()
    print(f"{'Norm of Primal residual:':25}{np.linalg.norm(
        A.T @ y + G.T @ w + q - lam + c
    ):<8.2e}")
    print(f"{'Norm of Dual residual:':25}{np.linalg.norm(np.concat((
            A @ x + z1 - b,
            Gx - h,
            x + z2 - u,
            -x + z3
        ))):<8.2e}")
    print(f"{'Abs. Compl. Gap:':25}{r_c:<8.2e}")
    scale = 1.0 + abs(primal_obj) + abs(dual_obj)
    print(f"{'Rel. Compl. Gap:':25}{r_c / scale:<8.2e}")
    print()
    if verbose:
        print(x)
        print(test)
    return -dual_obj, run_time, x, r_c

if __name__ == "__main__":
    ipm_solve_no_split_fo(verbose=True)