from __future__ import annotations

from ipm.problem_to_eq_ineq_matrices import problem_to_eq_ineq_matrices
from ipm.ipm_utility import ratio
import numpy as np
import scipy
import time

from pymatting import ichol

# Solve a GasNet problem in a relaxed LP form.
def ipm_solve_so(f_name: str = "linear_approx.json",
                xi: float = 1 - 5e-8, omega: float = 1e4,
                w_update: bool = True, D_bound: float = 1 - 5e-8, D_mult: float = .9,
                tau_init: float = 1e-4, scaling_tau: bool = False,
                epsilon: float = 1e-8, zeta_n: float = 5e-2, zeta_r: float = .5,
                alpha_sched: list[float] = [.9, .7, .5, .3, .1, .01], alpha_min: float = 1e-3,
                saund_tom: bool = False, saund_tom_factor: float = 1e-16,
                sparse: bool = False, classic_ilu: bool = False,
                refinement_iters: int = 3, ilu_drop_tol: float = 1e-4,
                omega_bound: bool = True, sym_neighborhood: bool = True,
                infeas_factor: float = 2.0, verbose: bool = False):
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
    m = i_bar + 2*e_bar + 2*n  # Number of constraints

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
    y = np.ones(i_bar) * omega
    w1 = np.ones(e_bar) * omega
    w2 = np.ones(e_bar) * omega
    q = np.ones(n) * omega
    lam = np.ones(n) * omega
    z1 = np.ones(i_bar) * omega
    z2 = np.ones(e_bar) * omega
    z3 = np.ones(e_bar) * omega
    z4 = np.ones(n) * omega
    z5 = np.ones(n) * omega

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
            A.T @ y + G.T @ (w1 - w2) + q - lam + c
        )
        r_d_norm = np.linalg.norm(np.concat((
            A @ x + z1 - b,
            Gx + z2 - h,
            -Gx + z3 + h,
            x + z4 - u,
            -x + z5
        )))
        r_c = np.dot(y, z1) + np.dot(w1, z2) + np.dot(w2, z3) \
            + np.dot(q, z4) + np.dot(lam, z5)

        if r_c <= epsilon:
            print("Solution is below target precision.")
            break

        y_1 = np.reciprocal(y)
        w1_1 = np.reciprocal(w1)
        w2_1 = np.reciprocal(w2)
        q_1 = np.reciprocal(q)
        lam_1 = np.reciprocal(lam)
        z1_1 = np.reciprocal(z1)
        z2_1 = np.reciprocal(z2)
        z3_1 = np.reciprocal(z3)
        z4_1 = np.reciprocal(z4)
        z5_1 = np.reciprocal(z5)

        if saund_tom:
            yz1_1 = 1 / (z1 * y_1 + saund_tom_factor)
            w1z2_1 = 1 / (z2 * w1_1 + saund_tom_factor)
            w2z3_1 = 1 / (z3 * w2_1 + saund_tom_factor)
            qz4_1 = 1 / (z4 * q_1 + saund_tom_factor)
            lamz5_1 = 1 / (z5 * lam_1 + saund_tom_factor)
        else:
            yz1_1 = y * z1_1
            w1z2_1 = w1 * z2_1
            w2z3_1 = w2 * z3_1
            qz4_1 = q * z4_1
            lamz5_1 = lam * z5_1 

        if sparse:
            ATZ1_1Y = A.T.multiply(yz1_1)
            GTZ_1W = G.T.multiply(w1z2_1 + w2z3_1)
        else:
            ATZ1_1Y = A.T * yz1_1
            GTZ_1W = G.T * (w1z2_1 + w2z3_1)

        ATy = A.T @ y
        GTw1_w2 = G.T @ (w1 - w2)

        if sparse:
            M = (ATZ1_1Y @ A + GTZ_1W @ G
                + scipy.sparse.diags_array(qz4_1 + lamz5_1)).tocsc()
            if saund_tom:
                M += saund_tom_factor * scipy.sparse.eye(M.shape[0], format="csc")
        else:
            M = (ATZ1_1Y @ A + GTZ_1W @ G
                + np.diag(qz4_1 + lamz5_1))
            if saund_tom:
                M += saund_tom_factor * np.eye(M.shape[0])

        # Predictor Step
        r_aff = ATZ1_1Y @ b_Ax + GTZ_1W @ h_Gx + qz4_1 * u_x - lamz5_1 * x \
            - c - ATy - GTw1_w2 - q + lam

        if classic_ilu:
            M_csc = scipy.sparse.csc_matrix(M)
            M_ichol = ichol(
                M_csc, discard_threshold=ilu_drop_tol
            )
            delta_x_aff = M_ichol(r_aff)
            if verbose:
                residual = r_aff - M_csc @ delta_x_aff
                residual_norm = np.linalg.norm(residual)
                print(
                    f"Before refinement residual norm: {residual_norm}"
                )
            for refine_step in range(refinement_iters):
                residual = r_aff - M_csc @ delta_x_aff
                correction = M_ichol(residual)
                candidate = delta_x_aff + correction
                candidate_residual = r_aff - M_csc @ candidate
                residual_norm = np.linalg.norm(residual)
                candidate_norm = np.linalg.norm(candidate_residual)
                if not np.isfinite(candidate_norm) or candidate_norm >= residual_norm:
                    break
                delta_x_aff = candidate
                if verbose:
                    print(
                        f"Refinement iteration {refine_step + 1} "
                        f"residual norm: {candidate_norm}"
                    )
        elif sparse:
            M_solver = scipy.sparse.linalg.factorized(M)
            delta_x_aff = M_solver(r_aff)
        else:
            M_lu, M_piv = scipy.linalg.lu_factor(M)
            delta_x_aff = scipy.linalg.lu_solve((M_lu, M_piv), r_aff)

        # Recover the steps
        Adelta_x_aff = A @ delta_x_aff
        Gdelta_x_aff = G @ delta_x_aff

        delta_z1_aff = b - Ax - z1 - Adelta_x_aff
        delta_z2_aff = h - Gx - z2 - Gdelta_x_aff
        delta_z3_aff = Gx - h - z3 + Gdelta_x_aff
        delta_z4_aff = u - x - z4 - delta_x_aff
        delta_z5_aff = x - z5 + delta_x_aff

        delta_y_aff = -y * z1_1 * delta_z1_aff - y
        delta_w1_aff = -w1 * z2_1 * delta_z2_aff - w1
        delta_w2_aff = -w2 * z3_1 * delta_z3_aff - w2
        delta_q_aff = -q * z4_1 * delta_z4_aff - q
        delta_lam_aff = c + ATy + GTw1_w2 + q - lam \
            + A.T @ delta_y_aff + G.T @ (delta_w1_aff - delta_w2_aff) + delta_q_aff

        alpha_star_y_aff = ratio(y, delta_y_aff)
        alpha_star_w1_aff = ratio(w1, delta_w1_aff)
        alpha_star_w2_aff = ratio(w2, delta_w2_aff)
        alpha_star_q_aff = ratio(q, delta_q_aff)
        alpha_star_lam_aff = ratio(lam, delta_lam_aff)
        alpha_star_z1_aff = ratio(z1, delta_z1_aff)
        alpha_star_z2_aff = ratio(z2, delta_z2_aff)
        alpha_star_z3_aff = ratio(z3, delta_z3_aff)
        alpha_star_z4_aff = ratio(z4, delta_z4_aff)
        alpha_star_z5_aff = ratio(z5, delta_z5_aff)

        if verbose:
            print()
            print(f"alpha_star_y_aff = {alpha_star_y_aff}")
            print(f"alpha_star_w1_aff = {alpha_star_w1_aff}")
            print(f"alpha_star_w2_aff = {alpha_star_w2_aff}")
            print(f"alpha_star_q_aff = {alpha_star_q_aff}")
            print(f"alpha_star_lam_aff = {alpha_star_lam_aff}")
            print(f"alpha_star_z1_aff = {alpha_star_z1_aff}")
            print(f"alpha_star_z2_aff = {alpha_star_z2_aff}")
            print(f"alpha_star_z3_aff = {alpha_star_z3_aff}")
            print(f"alpha_star_z4_aff = {alpha_star_z4_aff}")
            print(f"alpha_star_z5_aff = {alpha_star_z5_aff}")
            print()

        alpha_max_aff = min(alpha_star_y_aff, alpha_star_w1_aff,
                alpha_star_w2_aff, alpha_star_q_aff, alpha_star_lam_aff,
                alpha_star_z1_aff, alpha_star_z2_aff, alpha_star_z3_aff,
                alpha_star_z4_aff, alpha_star_z5_aff, 1.0)

        y_pred = y + alpha_max_aff * delta_y_aff
        w1_pred = w1 + alpha_max_aff * delta_w1_aff
        w2_pred = w2 + alpha_max_aff * delta_w2_aff
        q_pred = q + alpha_max_aff * delta_q_aff
        lam_pred = lam + alpha_max_aff * delta_lam_aff
        z1_pred = z1 + alpha_max_aff * delta_z1_aff
        z2_pred = z2 + alpha_max_aff * delta_z2_aff
        z3_pred = z3 + alpha_max_aff * delta_z3_aff
        z4_pred = z4 + alpha_max_aff * delta_z4_aff
        z5_pred = z5 + alpha_max_aff * delta_z5_aff

        r_c_pred = np.dot(y_pred, z1_pred) + np.dot(w1_pred, z2_pred) + np.dot(w2_pred, z3_pred) \
            + np.dot(q_pred, z4_pred) + np.dot(lam_pred, z5_pred)

        mu_corr = (r_c_pred / r_c)**3 * (r_c / m)

        # Corrector step
        ATZ1_1 = A.T @ z1_1
        r_cent = -mu_corr * (ATZ1_1 + G.T @ (z2_1 - z3_1) + z4_1 - z5_1) \
                + A.T @ (z1_1 * delta_y_aff * delta_z1_aff) \
                + G.T @ (z2_1 * delta_w1_aff * delta_z2_aff - z3_1 * delta_w2_aff * delta_z3_aff)\
                + z4_1 * (delta_q_aff * delta_z4_aff) - z5_1 * (delta_lam_aff * delta_z5_aff)

        if classic_ilu:
            delta_x_cent = M_ichol(r_cent)
            if verbose:
                residual = r_cent - M_csc @ delta_x_cent
                residual_norm = np.linalg.norm(residual)
                print(
                    f"Before refinement residual norm: {residual_norm}"
                )
            for refine_step in range(refinement_iters):
                residual = r_cent - M_csc @ delta_x_cent
                correction = M_ichol(residual)
                candidate = delta_x_cent + correction
                candidate_residual = r_cent - M_csc @ candidate
                residual_norm = np.linalg.norm(residual)
                candidate_norm = np.linalg.norm(candidate_residual)
                if not np.isfinite(candidate_norm) or candidate_norm >= residual_norm:
                    break
                delta_x_cent = candidate
                if verbose:
                    print(
                        f"Refinement iteration {refine_step + 1} "
                        f"residual norm: {candidate_norm}"
                    )
        elif sparse:
            delta_x_cent = M_solver(r_cent)
        else:
            delta_x_cent = scipy.linalg.lu_solve((M_lu, M_piv), r_cent)

        Adelta_x_cent = A @ delta_x_cent
        Gdelta_x_cent = G @ delta_x_cent

        delta_z1_cent = -Adelta_x_cent
        delta_z2_cent = -Gdelta_x_cent
        delta_z3_cent = Gdelta_x_cent
        delta_z4_cent = -delta_x_cent
        delta_z5_cent = delta_x_cent

        delta_y_cent = (-(delta_y_aff * delta_z1_aff) - y * delta_z1_cent + mu_corr) * z1_1
        delta_w1_cent = (-(delta_w1_aff * delta_z2_aff) - w1 * delta_z2_cent + mu_corr) * z2_1
        delta_w2_cent = (-(delta_w2_aff * delta_z3_aff) - w2 * delta_z3_cent + mu_corr) * z3_1
        delta_q_cent = (-(delta_q_aff * delta_z4_aff)  - q * delta_z4_cent + mu_corr) * z4_1
        delta_lam_cent = A.T @ delta_y_cent + G.T @ (delta_w1_cent - delta_w2_cent) + delta_q_cent

        delta_x = delta_x_aff + delta_x_cent
        delta_w1 = delta_w1_aff + delta_w1_cent
        delta_w2 = delta_w2_aff + delta_w2_cent
        delta_y = delta_y_aff + delta_y_cent
        delta_q = delta_q_aff + delta_q_cent
        delta_lam = delta_lam_aff + delta_lam_cent
        delta_z1 = delta_z1_aff + delta_z1_cent
        delta_z2 = delta_z2_aff + delta_z2_cent
        delta_z3 = delta_z3_aff + delta_z3_cent
        delta_z4 = delta_z4_aff + delta_z4_cent
        delta_z5 = delta_z5_aff + delta_z5_cent

        alpha_star_y = ratio(y, delta_y)
        alpha_star_w1 = ratio(w1, delta_w1)
        alpha_star_w2 = ratio(w2, delta_w2)
        alpha_star_q = ratio(q, delta_q)
        alpha_star_lam = ratio(lam, delta_lam)
        alpha_star_z1 = ratio(z1, delta_z1)
        alpha_star_z2 = ratio(z2, delta_z2)
        alpha_star_z3 = ratio(z3, delta_z3)
        alpha_star_z4 = ratio(z4, delta_z4)
        alpha_star_z5 = ratio(z5, delta_z5)

        if verbose:
            print()
            print(f"alpha_star_y = {alpha_star_y}")
            print(f"alpha_star_w1 = {alpha_star_w1}")
            print(f"alpha_star_w2 = {alpha_star_w2}")
            print(f"alpha_star_q = {alpha_star_q}")
            print(f"alpha_star_lam = {alpha_star_lam}")
            print(f"alpha_star_z1 = {alpha_star_z1}")
            print(f"alpha_star_z2 = {alpha_star_z2}")
            print(f"alpha_star_z3 = {alpha_star_z3}")
            print(f"alpha_star_z4 = {alpha_star_z4}")
            print(f"alpha_star_z5 = {alpha_star_z5}")
            print()

        alpha_max = min(alpha_star_y, alpha_star_w1, alpha_star_w2,
                        alpha_star_q, alpha_star_lam, alpha_star_z1,
                        alpha_star_z2, alpha_star_z3, alpha_star_z4,
                        alpha_star_z5, 1.0)

        is_neighbor = False

        for alpha_factor in alpha_sched:
            alpha = alpha_factor * alpha_max
            if alpha <= alpha_min:
                break

            x_alpha = x + alpha * delta_x
            y_alpha = y + alpha * delta_y
            w1_alpha = w1 + alpha * delta_w1
            w2_alpha = w2 + alpha * delta_w2
            q_alpha = q + alpha * delta_q
            lam_alpha = lam + alpha * delta_lam
            z1_alpha = z1 + alpha * delta_z1
            z2_alpha = z2 + alpha * delta_z2
            z3_alpha = z3 + alpha * delta_z3
            z4_alpha = z4 + alpha * delta_z4
            z5_alpha = z5 + alpha * delta_z5

            r_c_alpha = np.dot(y_alpha, z1_alpha) + np.dot(w1_alpha, z2_alpha) \
                + np.dot(w2_alpha, z3_alpha) + np.dot(q_alpha, z4_alpha) + np.dot(lam_alpha, z5_alpha)
            mu_alpha = r_c_alpha / m

            r_c_vec = np.concat((y_alpha * z1_alpha, w1_alpha * z2_alpha, w2_alpha * z3_alpha,
                                    q_alpha * z4_alpha, lam_alpha * z5_alpha))

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
                A.T @ y_alpha + G.T @ (w1_alpha - w2_alpha) + q_alpha - lam_alpha + c
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
                Gx_alpha + z2_alpha - h,
                -Gx_alpha + z3_alpha + h,
                x_alpha + z4_alpha - u,
                -x_alpha + z5_alpha
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
        w1 = w1_alpha
        w2 = w2_alpha
        q = q_alpha
        lam = lam_alpha
        z1 = z1_alpha
        z2 = z2_alpha
        z3 = z3_alpha
        z4 = z4_alpha
        z5 = z5_alpha

        # Adjust the w variables
        # print(f"w1 before update: {w1}")
        # print(f"w2 before update: {w2}")

        if scaling_tau:
            tau = tau_init * mu_corr
        else:
            tau = tau_init

        if w_update:
            D = np.maximum(tau, np.minimum(w1, w2) - tau)
            alpha_D = min(1, np.min((w1 * z2 - (zeta_n * mu_alpha)) / (D * z2)),
                np.min((w2 * z3 - (zeta_n * mu_alpha)) / (D * z3)))

            # Ensure we don't reach a centrality boundary
            # Can change the bound or the multiplier 
            if alpha_D < D_bound:
                alpha_D *= D_mult

            w1 -= alpha_D * D
            w2 -= alpha_D * D

        # print(f"w1 after update: {w1}")
        # print(f"w2 after update: {w2}")

        # If the problem is called infeasible, try raising infeas_factor.
        if max(np.max(np.abs(values), initial=0.0) for values in
               (x, y, w1, w2, q, lam, z1, z2, z3, z4, z5)) > infeas_factor * m * omega:
            print("The problem is infeasible.")
            break

        iteration += 1

        Gx = G @ x

        if verbose:
            r_c = np.dot(y, z1) + np.dot(w1, z2) + np.dot(w2, z3) \
                + np.dot(q, z4) + np.dot(lam, z5)
            print(f"Iteration {iteration}:")
            primal_obj = np.dot(b, y) + np.dot(h, w1 - w2) + np.dot(u, q) - const
            dual_obj = np.dot(-c, x) - const  # pylint: disable=invalid-unary-operand-type
            print(f"{'Primal objective:':25}{primal_obj:<15.8e}")
            print(f"{'Dual objective:':25}{dual_obj:<15.8e}")
            print()
            print(f"{'Primal residual:':25}{np.linalg.norm(
                A.T @ y + G.T @ (w1 - w2) + q - lam + c
            ):<8.2e}")
            print(f"{'Dual residual:':25}{np.linalg.norm(np.concat((
                    A @ x + z1 - b,
                    Gx + z2 - h,
                    -Gx + z3 + h,
                    x + z4 - u,
                    -x + z5
                ))):<8.2e}")
            print(f"{'Abs. Compl. Gap:':25}{r_c:<8.2e}")
            scale = 1.0 + abs(primal_obj) + abs(dual_obj)
            print(f"{'Rel. Compl. Gap:':25}{r_c / scale:<8.2e}")
            print()

    run_time = time.time() - start_time
    primal_obj = np.dot(b, y) + np.dot(h, w1 - w2) + np.dot(u, q) - const
    dual_obj = np.dot(-c, x) - const  # pylint: disable=invalid-unary-operand-type
    r_c = np.dot(y, z1) + np.dot(w1, z2) + np.dot(w2, z3) \
        + np.dot(q, z4) + np.dot(lam, z5)
    print(f"The algorithm stopped after {iteration:d} iterations in {run_time:.2f} seconds.")
    print()
    print(f"{'Primal objective:':25}{primal_obj:<15.8e}")
    print(f"{'Dual objective:':25}{dual_obj:<15.8e}")
    print()
    print(f"{'Norm of Primal residual:':25}{np.linalg.norm(
        A.T @ y + G.T @ (w1 - w2) + q - lam + c
    ):<8.2e}")
    print(f"{'Norm of Dual residual:':25}{np.linalg.norm(np.concat((
            A @ x + z1 - b,
            Gx + z2 - h,
            -Gx + z3 + h,
            x + z4 - u,
            -x + z5
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
    ipm_solve_so(verbose=True)