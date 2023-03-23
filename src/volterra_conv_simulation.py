"""
The module provides an universal simulation method for Volterra convolutions with
time independent coefficients.
"""

import numpy as np
from typing import Union, Callable
from numpy.typing import NDArray
from numpy import float_
from scipy.special import hyp2f1
from scipy.signal import convolve


def simulate_stochastic_convolution(
    T: float,
    n: int,
    alpha: float,
    y0: float,
    mu: Union[float, Callable[[Union[float, NDArray[float_]]], Union[float, NDArray[float_]]]],
    sigma: Union[float, Callable[[Union[float, NDArray[float_]]], Union[float, NDArray[float_]]]],
    kappa: int = 3,
    is_positive: bool = False
) -> NDArray[float_]:
    """
    Simulates stochastic convolution process
    dY_t = (t)^alpha (*) (mu(Y_t) dt + sigma(Y_t) dW_t), Y_0 = y0,
    where (*) stands for convolution.

    Args:
        T: the end of the time interval.
        n: number of discretization points per unit time.
        alpha: roughness parameter. alpha = H - 0.5, where H is Hurst exponent.
        y0: value of the process at t = 0.
        mu: trend from process Volterra SDE. Can be constant or callable function of Y_t.
        sigma: volatility from process Volterra SDE. Can be constant or callable function of Y_t.
        kappa: parameter of algorithm, number of intervals for which more accurate approximation is used.
        is_positive: whether the trajectory should be positive.

    Returns:
        Y: simulated trajectory on the interval [0, T].
        dW: inrements of the driving Brownian motions.
    """
    if callable(mu) ^ callable(sigma):
        raise ValueError("Both mu and sigma should be either functions or constants")

    # interval length in points
    N = int(np.floor(T * n) + 1)
    # first N elements will be zero to ease vectorization
    sigma_discr = np.zeros(2 * N)
    mu_discr = np.zeros(2 * N)
    Y = y0 * np.ones(N)

    cov_mat = np.zeros((kappa + 1, kappa + 1))
    cov_mat[0, 0] = 1 / n
    cov_mat[0, 1:] = cov_mat[1:, 0] = (np.arange(1, kappa + 1)**(alpha + 1) - np.arange(0, kappa)**(alpha + 1)) / ((alpha + 1) * n**(alpha + 1))
    np.fill_diagonal(cov_mat[1:, 1:], (np.arange(1, kappa + 1)**(2 * alpha + 1) - np.arange(0, kappa)**(2 * alpha + 1)) / ((2 * alpha + 1) * n**(2 * alpha + 1)))
    for i in range(1, kappa + 1):
        for k in range(i + 1, kappa + 1):
            a = i**(alpha + 1) * k**alpha * hyp2f1(-alpha, 1, alpha + 2, i / k)
            b = (i - 1)**(alpha + 1) * (k - 1)**alpha * hyp2f1(-alpha, 1, alpha + 2, (i - 1) / (k - 1))
            cov_mat[i, k] = (a - b) / ((alpha + 1) * n**(2 * alpha + 1))
            cov_mat[k, i] = cov_mat[i, k]

    # the first component of the gaussian vector correspons to the Wiener process increment on the interval.
    # the components 1:kappa correspond to the stochastic integrals of kernel with different t.
    W_matrix = np.random.multivariate_normal(mean=np.zeros(kappa + 1),
                                             cov=cov_mat,
                                             size=N)
    # first N elements will be zero to ease vectorization
    dW = np.zeros(2 * N)
    dW[N:] = W_matrix[:, 0]
    W_shifted = np.zeros((N + kappa - 1, kappa))
    for i in range(kappa):
        W_shifted[i:i + N, i] = W_matrix[:, i + 1]

    b_star = ((np.arange(1, N + 1)**(alpha + 1) - np.arange(0, N)**(alpha + 1)) / (alpha + 1))**(1 / alpha)
    G = (b_star / n)**alpha
    G[:kappa + 1] = 0
    G_mu = (np.arange(1, N)**(alpha + 1) - np.arange(N - 1)**(alpha + 1)) / (alpha + 1) / n**(alpha + 1)

    # # stupid implementation for tests. Actually helps to understand what's happening
    # for i in range(N):
    #     for k in range(1, i+1):
    #         Y[i] += mu_discr[N + i - k] * ((k / n)**(alpha + 1) - ((k-1) / n)**(alpha + 1)) / (alpha + 1)
    #         if k <= kappa:
    #             Y[i] += sigma_discr[N + i - k] * W_matrix[i - k, k]
    #         else:
    #             Y[i] += sigma_discr[N + i - k] * W_matrix[i - k, 0] * (b_star[k] / n)**alpha
    #     Y[i] = max(Y[i], 0)
    #     sigma_discr[N + i] = sigma(Y[i])
    #     mu_discr[N + i] = mu(Y[i])

    if callable(mu):
        sigma_discr[N] = sigma(y0)
        mu_discr[N] = mu(y0)
        for i in range(1, N):
            Y[i] += np.sum(mu_discr[N + i - 1: N + i - N: -1] * G_mu)
            Y[i] += np.sum(sigma_discr[N + i - 1: N + i - 1 - kappa: -1] * W_shifted[i - 1, :])
            Y[i] += np.sum(sigma_discr[N + i - kappa - 1: N + i - N: -1] * dW[N + i - kappa - 1: N + i - N: -1] * G[kappa + 1:])
            if is_positive:
                Y[i] = max(Y[i], 0)
            sigma_discr[N + i] = sigma(Y[i])
            mu_discr[N + i] = mu(Y[i])
    else:
        Y[1:] += sigma * np.sum(W_shifted[:N - 1, :], axis=1)
        Y += sigma * convolve(G, dW[N:], mode="full", method="fft")[:N]
        Y += mu * (np.arange(N) / n)**(alpha + 1) / (alpha + 1)
        if is_positive:
            Y = np.maximum(Y, 0)
    return Y, W_matrix[:-1, 0]
