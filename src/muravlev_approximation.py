import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, beta, roots_jacobi
from scipy.integrate import simps

def fbm_muravlev(n: int, hurst: float, mode: str = "standard", n_beta: float = 2000, beta_max: float = 2000,
                 uniform_beta_grid: bool = True, plotting: bool = False, extended_return: bool = False):
    """
        Simulates one trajectory of the fBM on the uniform grid on [0, 1].

        Args:
            n: number of equispaced increments desired for a fBM simulation.
            hurst: hurst exponent of the fBM.
            mode: "standard" to use integral from Muravlev's article. "gauss-jacobi" to use integral with limits (-1, 1) and approximate it by Gauss-Jacobi quadrature
            n_beta: size of the integral discretization grid.
            beta_max: the value of beta at which the integral is truncated. Used only for 'mode' == "standard".
            uniform_beta_grid: whether the beta grid is uniform. Used only for 'mode' == "standard".
            plotting: whether to plot the initial values xi and trajectories of OU processes. Used only for 'mode' == "standard".
            extended_return: if True, returns fBM and additional processes: OU processes in case if 'mode' == "standard",
                and integrand process and its mean for 'mode' == "gauss-jacobi"

        Returns:
            fBM: Trajectory of the fBM (np.ndarray of size n+1) approximation if 'extended_return' == False.
            (fBM, Z): Trajectory of the fBM (np.ndarray of shape (n+1,)) approximation and OU processes (np.ndarray of shape (n+1, n_beta))
                if 'extended_return' == True and 'mode' == "standard".
            (fBM, Y, eta, alphas): Trajectory of the fBM (np.ndarray of shape (n+1,)) approximation, integrands Y'' (np.ndarray of shape (n+1, n_beta)),
                its means 'eta' and quadrature points 'alphas' if 'extended_return' == True and 'mode' == "gauss-jacobi".
    """

    # ToDo: add T as a function argument and use self-similarity to transfrom trajectory generated on [0, 1] to the trajectory on [0, T].
    T = 1

    if mode == "standard":
        # Time grid and beta grid
        time_grid = np.linspace(0, 1, n + 1)
        if uniform_beta_grid:
            betas = np.linspace(beta_max / n_beta, beta_max, n_beta)
        else:
            betas = np.logspace(-1, np.log10(beta_max), n_beta)

        # Covariance matrix of the process xi
        xi_cov = 1 / (betas[:, None] + betas[None, :])
        # Generate xi as multivariate Gaussian r.v. - the most time-consuming part of code!
        xi = np.random.multivariate_normal(mean=np.zeros_like(betas), cov=xi_cov, size=1)[0]

        # Trajectories of the OU processes via Euler's discretization
        dt = T / n
        dW = np.random.randn(n) * np.sqrt(dt)
        Z = np.zeros((n + 1, n_beta))
        Z[0] = xi
        for i in range(n):
            Z[i + 1] = (1 - betas*dt)*Z[i] + dW[i]

        # Trajectory of FBM
        const_H = np.sqrt(gamma(2*hurst + 1) * np.sin(np.pi * hurst)) / beta(0.5 + hurst, 0.5 - hurst)
        fBM = const_H * simps(betas**(-0.5 - hurst) * (Z - xi), betas)

        if plotting:
            _, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(betas, xi)
            ax[0].set_title("Process xi")
            ax[0].grid()

            n_ou_trajs = 10
            for j in range(n_ou_trajs):
                z_index = (2 * j)**2
                if z_index < len(betas):
                    ax[1].plot(time_grid, Z[:, z_index], label=r"$\beta = $" + str(np.round(betas[z_index], 1)))
            ax[1].set_title("Simulated trajectories of OUs")
            ax[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax[1].grid()

            _, ax_res = plt.subplots(figsize=(8, 5))
            ax_res.plot(time_grid, fBM)
            ax_res.set_title("Muravlev's approximation of fBM")
            ax_res.grid()
            plt.show()

        return fBM if not extended_return else (fBM, Z)

    if mode == "gauss-jacobi":
        a = hurst - 1
        b = -0.5 - hurst
        alphas, weights = roots_jacobi(n_beta, a, b)
        betas = (1 + alphas) / (1 - alphas)

        eta_cov = 0.5 * np.sqrt((1 - alphas[:, None])*(1 - alphas[None, :])) / (1 - alphas[:, None] * alphas[None, :])
        eta = np.random.multivariate_normal(mean=np.zeros_like(alphas), cov=eta_cov, size=1)[0]

        dt = 1 / n
        zeta = np.random.randn(n)[:, None] * np.sqrt(0.5 * (1 - np.exp(-2*betas*dt)) / (1 + alphas))[None, :]
        Y = np.zeros((n + 1, n_beta))
        for i in range(n):
            Y[i + 1] = np.exp(-betas*dt)*(Y[i] + eta) - eta + zeta[i]

        const_H = np.sqrt(gamma(2*hurst + 1) * np.sin(np.pi * hurst)) / beta(0.5 + hurst, 0.5 - hurst)
        fBM = 2 * const_H * Y @ weights

        return fBM if not extended_return else (fBM, Y, eta, alphas)

    return None
