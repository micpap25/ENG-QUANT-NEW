from __future__ import annotations

from ipm_new.problem_to_eq_ineq_matrices import problem_to_eq_ineq_matrices
import numpy as np
import time

# Solve a GasNet problem in a relaxed LP form.

def ratio(x_vec, delta_x_vec):
    rat = 1
    for (x, delta_x) in zip(x_vec, delta_x_vec):
        if delta_x < 0:
            rat = min(- x / delta_x , rat)
    return rat

# Times the cost so we have an upper bound on the objective value
def ipm_solve_toy(beta: float = 0.1, beta2: float = 1 - 5e-4, omega: float = 1e2,
                gamma: float = 0.05, precision: float = 1e-16,
                alpha_hat_dec: float = .9, step_precision: float = 1e-4) -> None:
    np.set_printoptions(linewidth=200)

    # Primal solution is (3, 1), obj = -9
    A = np.array([[1, 3], \
                  [2, 1],
                  [-1, 0],
                  [0, -1]])
    b = np.array([6, 7, -4, -4])
    c = np.array([-2, -3])
    u = np.array([100, 100])

    i = len(b)
    n = len(c)  # Number of variables
    m = i + 2 * n  # Number of constraints

    # problem-to-eq-ineq-matrix sanity check
    assert len(A) == i
    assert len(A[0]) == n
    assert len(u) == n

     # Run II-IPM
    x = np.ones(n) * omega
    y = -np.ones(i) * omega
    q = np.ones(n) * omega
    lam = np.ones(n) * omega
    z1 = np.ones(i) * omega
    z4 = np.ones(n) * omega

    iteration = 0
    start_time = time.time()

    while True:
        compl = -np.dot(y, z1) + np.dot(q, z4) + np.dot(x, lam)

        mu = compl * beta / m

        X = np.diag(x)
        Y = np.diag(y)
        Q = np.diag(q)
        Lam = np.diag(lam)
        Z1 = np.diag(z1)
        Z4 = np.diag(z4)

        ATZ_1 = np.dot(A.T, np.linalg.inv(Z1))
        ATZ_1Y = np.dot(ATZ_1, Y)
        Ax = np.dot(A, x)
        b_Ax = b - Ax
        u_x = u - x
        ATy = np.dot(A.T, y)
        Z4_1Q = np.dot(np.linalg.inv(Z4), Q)

        M = np.dot(ATZ_1Y, A) - Z4_1Q - np.dot(np.linalg.inv(X), Lam)

        r = np.dot(ATZ_1Y, b_Ax) - np.dot(Z4_1Q, u_x) \
            - (mu * np.dot(A.T, np.reciprocal(z1))) + (mu * np.reciprocal(z4)) \
            - (mu * np.reciprocal(x)) + ATy + c + q

        # Linear System Solve
        delta_x = np.linalg.solve(M, r)

        # Recover the steps
        Adelta_x = np.dot(A, delta_x)
        delta_y = np.dot(np.linalg.inv(Z1), np.dot(Y, b - Ax - Adelta_x) - (mu * np.ones(i)))
        delta_q = np.dot(np.linalg.inv(Z4), np.dot(Q, delta_x + x - u) + (mu * np.ones(n)))
        delta_lam = c + q + ATy - lam + np.dot(A.T, delta_y) + delta_q
        delta_z1 = -b + Ax - z1 + Adelta_x
        delta_z4 = u - x - z4 - delta_x

        alpha_star_x = ratio(x, delta_x)
        alpha_star_y = ratio(-y, -delta_y)
        alpha_star_q = ratio(q, delta_q)
        alpha_star_lam = ratio(lam, delta_lam)
        alpha_star_z1 = ratio(z1, delta_z1)
        alpha_star_z4 = ratio(z4, delta_z4)

        alpha_hat = min(alpha_star_x, alpha_star_y, alpha_star_q, alpha_star_lam,
                        alpha_star_z1, alpha_star_z4, alpha_hat_dec)

        # Calculate alpha_hat
        is_neighbor = False

        while not is_neighbor:
            x_temp = x + alpha_hat * delta_x
            y_temp = y + alpha_hat * delta_y
            q_temp = q + alpha_hat * delta_q
            lam_temp = lam + alpha_hat * delta_lam
            z1_temp = z1 + alpha_hat * delta_z1
            z4_temp = z4 + alpha_hat * delta_z4

            compl_temp = -np.dot(y_temp, z1_temp) + np.dot(q_temp, z4_temp) + np.dot(lam_temp, x_temp)
            mu_temp = compl_temp / m
            compl_vec = np.concat((-y_temp * z1_temp, q_temp * z4_temp, lam_temp * x_temp))

            is_neighbor = True

            for comp in compl_vec:
                if comp < gamma * mu_temp:
                    alpha_hat *= alpha_hat_dec
                    is_neighbor = False
                    break

            if not is_neighbor:
                continue

            # The residual should be the norm of all the constraint errors
            epsilon_primal = np.linalg.norm(np.dot(A.T, y_temp) + q_temp - lam_temp + c)

            if epsilon_primal > max(compl_temp/gamma, precision):
                alpha_hat *= alpha_hat_dec
                is_neighbor = False
                continue

            epsilon_dual = np.linalg.norm(np.concat((
                np.dot(A, x_temp) - z1_temp - b,
                x_temp + z4_temp - u)))

            if epsilon_dual > max(compl_temp/gamma, precision):
                alpha_hat *= alpha_hat_dec
                is_neighbor = False
                continue

            if compl_temp > (1 - alpha_hat * (1 - beta2)) * compl:
                alpha_hat *= alpha_hat_dec
                is_neighbor = False
                continue

        x = x_temp
        y = y_temp
        q = q_temp
        lam = lam_temp
        z1 = z1_temp
        z4 = z4_temp

        if max(abs(entry) for entry in np.concat((x, y, q, lam, z1, z4))) > 2 * m * omega:
            print("The problem is infeasible.")
            break

        iteration += 1

        compl = -np.dot(y, z1) + np.dot(q, z4) + np.dot(lam, x)

        if compl <= precision:
            print("Solution is below target precision.")
            break

        print(f"Iteration {iteration}:")
        print(f"{'Primal objective:':20}{np.dot(b, y) + np.dot(u, q):<15.8e}")
        print(f"{'Dual objective:':20}{np.dot(-c, x):<15.8e}")
        print()
        print(f"{'Primal residual:':20}{np.linalg.norm(np.dot(A.T, y) + q - lam + c):<8.2e}")
        print(f"{'Dual residual:':20}{np.linalg.norm(np.concat((
                np.dot(A, x) - z1 - b,
                x + z4 - u))):<8.2e}")
        print(f"{'Complementarity:':20}{compl:<8.2e}")
        print()

        print(f"x: {x}")
        print(f"y: {y}")
        print(f"q: {q}")
        print(f"lam: {lam}")
        print(f"z1: {z1}")
        print(f"z4: {z4}")

        if alpha_hat < step_precision:
            print("The solution quality is limited by the precision of the linear system solver.")
            break
    
    run_time = time.time() - start_time
    print(f"The algorithm stopped after {iteration:d} iterations in {run_time:.2f} seconds.")
    print()
    print(f"{'Primal objective:':20}{np.dot(b, y) + np.dot(u, q):<15.8e}")
    print(f"{'Dual objective:':20}{np.dot(-c, x):<15.8e}")
    print()
    print(f"{'Primal residual:':20}{np.linalg.norm(np.dot(A.T, y) + q - lam + c):<8.2e}")
    print(f"{'Dual residual:':20}{np.linalg.norm(np.concat((
            np.dot(A, x) - z1 - b,
            x + z4 - u))):<8.2e}")
    print(f"{'Complementarity:':20}{compl:<8.2e}")
    print()

    print(x)
    
if __name__ == "__main__":
    ipm_solve_toy()
