import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, beta

def fbm_muravlev(n: int, hurst: float, n_beta: float = 2000, beta_max: float = 2000, plotting: bool = False, extended_return: bool = False):
    """
        Simulates one trajectory of the fBM on the uniform grid on [0, 1].

        Args:
            n: number of equispaced increments desired for a fBM simulation.
            hurst: hurst exponent of the fBM.
            n_beta: size of the integral discretization grid.
            beta_max: the value of beta at which the integral is truncated
            plotting: whether to plot the initial values xi and trajectories of OU processes.
            extended_return: whether to return the resulting fBM, initial values xi and trajectories of OU processes.

        Returns:
            fBM: Trajectory of the fBM (np.ndarray of size n+1) approximation if 'extended_return == False'.
            (fBM, xi, Z): Trajectory of the fBM (np.ndarray of shape (n+1,)) approximation and OU processes (np.ndarray of shape (n+1, n_beta))
    """

    # ToDo: add T as a function argument and use self-similarity to transfrom trajectory generated on [0, 1] to the trajectory on [0, T].
    T = 1

    # Time grid and beta grid
    # ToDo: try non-uniform beta grid or variables transformation
    time_grid = np.linspace(0, 1, n + 1)
    betas = np.linspace(beta_max / n_beta, beta_max, n_beta)

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
    fBM = const_H * np.sum(betas**(-0.5 - hurst) * (Z - xi) * np.diff(betas, prepend=0), axis=1)

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
        ax_res.set_title("Fractional BM")
        ax_res.grid()
        plt.show()

    return fBM if not extended_return else (fBM, Z)
