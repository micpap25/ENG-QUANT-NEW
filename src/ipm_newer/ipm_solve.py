from __future__ import annotations

from ipm_newer.problem_to_eq_ineq_matrices import problem_to_eq_ineq_matrices
import numpy as np
import time

import scipy.linalg
import scipy.sparse

import warnings

warnings.filterwarnings('error', category=scipy.sparse.linalg.MatrixRankWarning)

# Solve a GasNet problem in a relaxed LP form.

def ratio(x_vec, delta_x_vec):
    rat = 1
    for (x, delta_x) in zip(x_vec, delta_x_vec):
        if delta_x < 0:
            rat = min(- x / delta_x , rat)
    return rat

def ipm_solve(f_name: str = "linear_approx.json", beta: float = .1,
                xi: float = 1 - 5e-4, omega: float = 1e4,
                w_update: bool = True, D_bound: float = 1 - 5e-8, D_mult: float = .9,
                tau_init: float = 1e-4, scaling_tau: bool = False,
                epsilon: float = 1e-8, zeta_n: float = 5e-2, zeta_r: float = .8,
                alpha_sched: list[float] = [.9, .7, .5, .3, .1, .01], alpha_min: float = 1e-3,
                saund_tom: bool = False, saund_tom_factor: float = 0,
                sparse: bool = True, ilu: bool = False,
                ilu_refinement: int = 2, ilu_drop_tol: float = 1e-10,
                verbose: bool = False):
    np.set_printoptions(linewidth=200)

    # Gurobi output pops up here because of license file
    A, b, G, h, u, c, const = problem_to_eq_ineq_matrices(f_name=f_name, sparse=sparse, verbose=False)

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
    max_cost = np.dot(np.clip(c, 0, None), u)
    omega = min(max_cost, omega)
    if verbose:
        print(max_cost)
        if omega == max_cost:
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

        mu = r_c * beta / m

        if verbose:
            print(f"mu: {mu}")

        if scaling_tau:
            tau = tau_init * mu
        else:
            tau = tau_init

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

        r_aff = ATZ1_1Y @ b_Ax + GTZ_1W @ h_Gx + qz4_1 * u_x - lamz5_1 * x \
            - c - ATy - GTw1_w2 - q + lam
        r_cent = A.T @ z1_1 + G.T @ (z2_1 - z3_1) + z4_1 - z5_1

        r = r_aff - mu * r_cent

        if ilu:
            M_csc = scipy.sparse.csc_matrix(M)
            M_ilu = scipy.sparse.linalg.spilu(
                M_csc, drop_tol=ilu_drop_tol
            )
            delta_x = M_ilu.solve(r)
            if verbose:
                residual = r - M_csc @ delta_x
                residual_norm = np.linalg.norm(residual)
                print(
                    f"Before refinement residual norm: {residual_norm}"
                )
            for refine_step in range(ilu_refinement):
                residual = r - M_csc @ delta_x
                correction = M_ilu.solve(residual)
                candidate = delta_x + correction
                candidate_residual = r - M_csc @ candidate
                residual_norm = np.linalg.norm(residual)
                candidate_norm = np.linalg.norm(candidate_residual)
                if not np.isfinite(candidate_norm) or candidate_norm >= residual_norm:
                    break
                delta_x = candidate
                if verbose:
                    print(
                        f"Refinement iteration {refine_step + 1} "
                        f"residual norm: {candidate_norm}"
                    )
        elif sparse:
            delta_x = scipy.sparse.linalg.spsolve(M, r)
        else:
            delta_x = np.linalg.solve(M, r)

        # Recover the steps
        Adelta_x = A @ delta_x
        Gdelta_x = G @ delta_x

        delta_z1 = b - Ax - z1 - Adelta_x
        delta_z2 = h - Gx - z2 - Gdelta_x
        delta_z3 = Gx - h - z3 + Gdelta_x
        delta_z4 = u - x - z4 - delta_x
        delta_z5 = x - z5 + delta_x

        delta_y = (-y * delta_z1 + mu) * z1_1 - y
        delta_w1 = (-w1 * delta_z2 + mu) * z2_1 - w1
        delta_w2 = (mu - w2 * delta_z3) * z3_1 - w2
        delta_q = (-q * delta_z4 + mu) * z4_1 - q
        delta_lam = c + ATy + GTw1_w2 + q - lam \
            + A.T @ delta_y + G.T @ (delta_w1 - delta_w2) + delta_q

        # Bad code for comparison
        # delta_y = (y * (Ax + Adelta_x - b) + mu) / z1
        # delta_w1 = (w1 * (Gx + Gdelta_x - h) + mu) / z2
        # delta_w2 = (mu - w2 * (Gx + Gdelta_x - h)) / z3
        # delta_q = (q * (x + delta_x - u) + mu) / z4
        # delta_lam = c + ATy + GTw1_w2 + q - lam \
        #     + A.T @ delta_y + G.T @ (delta_w1 - delta_w2) + delta_q

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

        alpha_max = min(alpha_star_y, alpha_star_w1,
                        alpha_star_w2, alpha_star_q, alpha_star_lam,
                        alpha_star_z1, alpha_star_z2, alpha_star_z3,
                        alpha_star_z4, alpha_star_z5, 1.0)

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

            for comp in r_c_vec:
                if comp < zeta_n * mu_alpha:
                    is_neighbor = False
                    test[0] += 1
                    break
                if comp > mu_alpha / zeta_n:
                    is_neighbor = False
                    test[6] += 1
                    break

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

        # Adjust the w variables (Is this better to do in the newton loop or is here OK?
        # No difference in actual convergence, seems identical)
        # print(f"w1 before update: {w1}")
        # print(f"w2 before update: {w2}")
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

        if max(abs(entry) for entry in np.concat(
            (x, y, w1, w2, q, lam, z1, z2, z3, z4, z5)
        )) > 2 * m * omega:
            print("The problem is infeasible.")
            break

        iteration += 1

        Gx = G @ x

        if verbose:
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
    ipm_solve(verbose=True)