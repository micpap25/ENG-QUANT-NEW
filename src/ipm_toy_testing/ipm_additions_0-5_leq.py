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
                zeta_n: float = 0.05, zeta_r: float = 0.05, precision: float = 1e-16,
                alpha_hat_dec: float = .9, step_precision: float = 1e-4) -> None:
    np.set_printoptions(linewidth=200)

    # Primal solution is (3, 1), obj = 9
    A = np.array([[1, 3],
                  [2, 1],
                  [-1, 0],
                  [0, -1]])
    b = np.array([6, 7, -1, -1])
    c = np.array([2, 3])

    i = len(b)
    n = len(c)  # Number of variables
    m = i + n  # Number of constraints

    # problem-to-eq-ineq-matrix sanity check
    assert len(A) == i
    assert len(A[0]) == n

    # Run II-IPM
    x = np.ones(n) * omega
    z = np.ones(i) * omega
    y = -np.ones(i) * omega
    s = np.ones(n) * omega

    rp_0 = np.linalg.norm(np.dot(A, x) + z - b)
    rd_0 = np.linalg.norm(np.dot(A.T, y) + s - c)
    mu_0 = omega ** 2

    iteration = 0
    start_time = time.time()

    while True:
        compl = np.dot(x, s) - np.dot(y, z)

        mu = compl * beta / m

        X = np.diag(x)
        Y = np.diag(y)
        S = np.diag(s)
        Z = np.diag(z)

        ATZ_1 = np.dot(A.T, np.linalg.inv(Z))
        ATZ_1Y = np.dot(ATZ_1, Y)
        Ax = np.dot(A, x)
        b_Ax = b - Ax
        ATy = np.dot(A.T, y)

        M = np.dot(ATZ_1Y, A) - np.dot(np.linalg.inv(X), S)

        r = np.dot(ATZ_1Y, b_Ax) + (mu * np.dot(A.T, np.reciprocal(z))) - (mu * np.reciprocal(x)) - ATy + c

        # Linear System Solve
        delta_x = np.linalg.solve(M, r)

        # Recover the steps
        Adelta_x = np.dot(A, delta_x)
        delta_z = b - Ax - z - Adelta_x
        delta_y = (-mu * np.ones(i) - y * (delta_z)) / z - y
        delta_s = c - ATy - s - np.dot(A.T, delta_y)

        alpha_star_x = ratio(x, delta_x)
        alpha_star_y = ratio(-y, -delta_y)
        alpha_star_s = ratio(s, delta_s)
        alpha_star_z = ratio(z, delta_z)

        alpha_hat = min(alpha_star_x, alpha_star_y, alpha_star_s, alpha_star_z, alpha_hat_dec)

        # Calculate alpha_hat
        is_neighbor = False

        while not is_neighbor:
            x_temp = x + alpha_hat * delta_x
            y_temp = y + alpha_hat * delta_y
            s_temp = s + alpha_hat * delta_s
            z_temp = z + alpha_hat * delta_z

            compl_temp = -np.dot(y_temp, z_temp) + np.dot(x_temp, s_temp)
            mu_temp = compl_temp / m
            mu_ratio = mu_temp / mu_0
            compl_vec = np.concat((-y_temp * z_temp, x_temp * s_temp))

            is_neighbor = True

            for comp in compl_vec:
                if comp < zeta_n * mu_temp:
                    alpha_hat *= alpha_hat_dec
                    is_neighbor = False
                    break

            if not is_neighbor:
                continue

            # The residual should be the norm of all the constraint errors
            epsilon_primal = np.linalg.norm(np.dot(A, x_temp) + z_temp - b)

            if epsilon_primal > max(rp_0 * mu_ratio/zeta_r, precision):
                alpha_hat *= alpha_hat_dec
                is_neighbor = False
                continue

            # if epsilon_primal < rp_0 * mu_ratio * zeta_r:
            #     alpha_hat *= alpha_hat_dec
            #     is_neighbor = False
            #     continue

            epsilon_dual = np.linalg.norm(np.dot(A.T, y_temp) + s_temp - c)

            if epsilon_dual > max(rd_0 * mu_ratio/zeta_r, precision):
                alpha_hat *= alpha_hat_dec
                is_neighbor = False
                continue

            # if epsilon_dual < rd_0 * mu_ratio * zeta_r:
            #     alpha_hat *= alpha_hat_dec
            #     is_neighbor = False
            #     continue

            if compl_temp > (1 - alpha_hat * (1 - beta2)) * compl:
                alpha_hat *= alpha_hat_dec
                is_neighbor = False
                continue

        x = x_temp
        y = y_temp
        s = s_temp
        z = z_temp

        if max(abs(entry) for entry in np.concat((x, y, s, z))) > 2 * m * omega:
            print("The problem is infeasible.")
            break

        iteration += 1

        compl = -np.dot(y, z) + np.dot(x, s)

        if compl <= precision:
            print("Solution is below target precision.")
            break

        print(f"Iteration {iteration}:")
        print(f"{'Primal objective:':20}{np.dot(c, x):<15.8e}")
        print(f"{'Dual objective:':20}{np.dot(b, y):<15.8e}")
        print()
        print(f"{'Primal residual:':20}{np.linalg.norm(np.dot(A, x) + z - b):<8.2e}")
        print(f"{'Dual residual:':20}{np.linalg.norm(np.dot(A.T, y) + s - c):<8.2e}")
        print(f"{'Complementarity:':20}{compl:<8.2e}")
        print()

        if alpha_hat < step_precision:
            print("The solution quality is limited by the precision of the linear system solver.")
            break
    
    run_time = time.time() - start_time
    print(f"The algorithm stopped after {iteration:d} iterations in {run_time:.2f} seconds.")
    print()
    print(f"{'Primal objective:':20}{np.dot(c, x):<15.8e}")
    print(f"{'Dual objective:':20}{np.dot(b, y):<15.8e}")
    print()
    print(f"{'Primal residual:':20}{np.linalg.norm(np.dot(A, x) + z - b):<8.2e}")
    print(f"{'Dual residual:':20}{np.linalg.norm(np.dot(A.T, y) + s - c):<8.2e}")
    print(f"{'Complementarity:':20}{compl:<8.2e}")
    print()

    print(x)
    
if __name__ == "__main__":
    ipm_solve_toy()
