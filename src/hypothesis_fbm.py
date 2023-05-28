import numpy as np
from scipy.linalg import sqrtm
from numpy.linalg import eigvals
from scipy.optimize import fsolve


def get_A_k(N, k):
    """
    Returns matrix of shape (N, N) with values 1 / (2 * (N - k)) on the k-th and -k-th diagonals.
    """
    A = np.diag(0.5*np.ones(N - k), k=k)
    A = (A + A.T) / (N - k)
    return A


def get_acvf(N, H, eta=0):
    """
    Returns r[k], k = 0, ..., N-1 - ACVF for fBM with Hurst exponent H + noise with variance eta.
    """
    k = np.arange(N)
    r = 0.5 * ((k + 1)**(2*H) + np.abs(k - 1)**(2*H) - 2*k**(2*H))

    r[0] = r[0] + 2*eta
    r[1] = r[1] - eta
    return r


def get_cov_mat(N, r):
    """
    Returns covatiance matrix of shape (N, N) with r[k] on k-th and -k-th diagonals, k = 0, ..., N-1.
    """
    cov_mat = np.zeros((N, N))

    for i in range(1, N):
        cov_mat += np.diag(r[i]*np.ones(N - i), i)

    cov_mat = cov_mat + cov_mat.T
    np.fill_diagonal(cov_mat, r[0])
    return cov_mat


def get_eigenvals(cov_mat, A):
    """
    Returns eigenvalues of matrix cov_mat**0.5 @ A @ cov_mat**0.5.
    """
    cov_mat_sqrt = sqrtm(cov_mat)
    C = cov_mat_sqrt @ A @ cov_mat_sqrt
    return eigvals(C)


def generalized_chi2_cf(lam):
    """
    Returns characteristic function (callable) of the generalized chi2 distribution
    lam[0]*U[0]**2 + ... + lam[-1]*U[-1]**2, where U[i] ~ N(0, 1).
    """
    return lambda t: np.prod((1 - 2*1j*lam.reshape((-1, 1)) @
                              np.array([t]).reshape((1, -1)))**(-0.5), axis=0).squeeze()


def generalized_chi2_cdf(lam, t_grid=None):
    """
    Returns cdf (callable) of the generalized chi2 distribution
    lam[0]*U[0]**2 + ... + lam[-1]*U[-1]**2, where U[i] ~ N(0, 1)
    computed via Gil-Pelaez theorem.

    Args:
        lam: distribution parameters, weights.
        t_grid: integration grid.
    """
    cf = generalized_chi2_cf(lam)
    if t_grid is None:
        t_grid = np.linspace(0, 100, 10**4) + 1e-9
    return lambda x: (0.5 - np.trapz((np.exp(-1j*np.array(x).reshape((-1, 1))@t_grid.reshape((1, -1))) *
                                     cf(t_grid)).imag / t_grid, t_grid) / np.pi).squeeze()


def generalized_chi2_quantile(lam, q):
    """
    Returns q-th quantile of the generalized chi2 distribution
    lam[0]*U[0]**2 + ... + lam[-1]*U[-1]**2, where U[i] ~ N(0, 1).
    """
    cdf = generalized_chi2_cdf(lam)
    return fsolve(func=lambda x: cdf(x) - q, x0=lam.sum())


def test_hypothesis(X, H, eta, alpha=0.05, k=1):
    """
    Tests the hypothesis that X is fBM with H with the noise variance eta.

    Args:
        X: process trajectory.
        H: Hurst exponent in H0 hypothesis.
        eta: variance of noise.
        alpha: significance level.
        k: parameter of the test, the statistic is the estimated ACVF r(k).

    Returns:
        (Test, Q, (q1, q2)): test is False iff H0 was rejected, Q is the value of the statistic,
        (q1, q2) is the interval between quantiles.
    """
    N = X.size
    A = get_A_k(N, k)
    r = get_acvf(N, H, eta)
    cov_mat = get_cov_mat(N, r)
    lam = get_eigenvals(cov_mat, A)
    q1, q2 = generalized_chi2_quantile(lam, 0.5*alpha), generalized_chi2_quantile(lam, 1 - 0.5*alpha)

    Q = X @ A @ X
    test = q1 <= Q <= q2

    return test[0], Q, (q1[0], q2[0])
