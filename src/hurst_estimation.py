"""
The module provides two estimators of the trajectory Hurst exponent, introduced in the articles
"Jim Gatheral, Thibault Jaisson, and Mathieu Rosenbaum. Volatility is rough. Quantitative Finance, 18:933 – 949, 2014."
and
"Rama Cont and Purba Das. Rough volatility: fact or artefact? https://arxiv.org/abs/2203.13820, 2022."
Main estimatimation methods are 'm_estimator' and 'w_estimator' respectively.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import root_scalar, minimize
from scipy.special import gamma, factorial

# Global ToDo:
# rewrite the roughness estimation methods in this model as classes.

# -----------------------------------------------------------------------
# ---      regression analysis from Gatheral, Jaisson, Rosenbaum      ---
# -----------------------------------------------------------------------


def my_diff(x, lag):
    """
    Calculates finite difference y of sequence x: y[t] = x[t] - x[t - lag]
    (x[t] are supposed to be 0 for negative t).

    Args:
        x: sequence.
        lag: difference step.

    Returns:
        The finite difference array of shape x.shape.
    """
    return x - np.concatenate([np.zeros(lag), x[:-lag]])


def get_m(X, q, delta, intersect=False):
    """
    Calculates the estimation of m from "Volatility is rough" for given step delta and q.

    Args:
        X: trajectory of the process.
        q: can be both np.ndarray and float.
        delta: integer step used in m.
        intersect: bool, whether the intervals (i, i+delta) intersect or not.

    Returns:
        The value of m of shape q.shape.
    """
    step = 1 if intersect else delta
    axis = 1 if type(q) == np.ndarray else None
    if type(q) == np.ndarray:
        q = q.reshape((q.size, 1))
    return (np.abs(my_diff(X, lag=delta)[delta::step])**q).mean(axis=axis)


def get_m_matrix(X, q, steps, intersect=False):
    """
    Calculates the estimation of m from "Volatility is rough" for given step delta and q for different delta (steps).

    Args:
        X: trajectory of the process.
        q: np.ndarray. contains values of powers q.
        steps: np.ndarray, contains values of delta.
        intersect: bool, whether the intervals (i, i+delta) intersect or not.

    Returns:
        The value of m of shape (q.size, steps.size).
    """
    m = np.zeros((q.size, steps.size))
    for i, delta in enumerate(steps):
        m[:, i] = get_m(X, q, delta, intersect=intersect)
    return m


def m_regression(m, q, steps, is_plotting=False, ax=None):
    """
    Builds the regression of log(m) over log(steps) for different values of q.

    Args:
        m: array of shape (q.size, steps.size).
        q: np.ndarray, contains values of powers q.
        steps: np.ndarray, contains values of delta.
        is_plotting: whether to plot the regression.

    Returns:
        Lists c, b of slopes and intercepts of size q.size.
    """
    if is_plotting and ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    c = np.zeros(m.shape[0])
    b = np.zeros(m.shape[0])

    for i, mq in enumerate(m):
        x = np.log(steps)
        y = np.log(mq)
        model = LinearRegression()
        model.fit(np.array([x]).T, y)
        c[i], b[i] = model.coef_[0], model.intercept_
        if is_plotting:
            ax.plot(np.log(steps), b[i] + c[i] * np.log(steps), 'k', lw=0.7)
            ax.scatter(np.log(steps), np.log(mq), label=f'q={q[i][0]}', s=20)

    if is_plotting:
        ax.set_xlabel(r'$\log(\Delta)$')
        ax.set_ylabel(r'$\log(m(q, \Delta))$')
        ax.legend()
        ax.grid()
    return c, b


def H_regression(q, c, is_plotting=False, ax=None):
    """
    Builds the regression of slope coefficientes c on q.

    Args:
        q: np.ndarray, contains values of powers q.
        c: np.ndarray, contains slope coefficients.
        is_plotting: whether to plot the regression.

    Returns:
        The slope coefficient H.
    """
    model = LinearRegression()
    model.fit(q, c)
    H, inter = model.coef_[0], model.intercept_

    if is_plotting:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        ax.plot(q.squeeze(), inter + H * q.squeeze(), 'k', lw=0.7)
        ax.scatter(q.squeeze(), c, color='g', s=70)

        ax.grid()
        ax.set_xlabel('q')
        ax.set_ylabel(r'$\zeta_q$')
    return H


def m_estimator(X, q=None, steps=None, is_plotting=False, intersect=False):
    """
    Provides the trajectory analysis from "Volatility is rough" for X.

    Args:
        X: trajectory of the process.
        q: np.ndarray, contains values of powers q.
        steps: np.ndarray, contains values of delta.
        is_plotting: whether to plot the regressions.
        intersect: bool, whether the intervals (i, i+delta) intersect or not.

    Returns:
        The m-stimation of Hurst exponent H.
    """
    if q is None:
        q = np.array([[0.5, 1, 1.5, 2, 2.5, 3]]).T
    if steps is None:
        steps = np.arange(1, 51)

    if is_plotting:
        _, ax = plt.subplots(1, 2, figsize=(15, 7))
    else:
        ax = [None, None]

    m = get_m_matrix(X, q, steps, intersect=intersect)
    c, _ = m_regression(m, q, steps, is_plotting, ax=ax[0])
    H_est = H_regression(q, c, is_plotting, ax=ax[1])
    return H_est

# -----------------------------------------------------------------------
# ---             Estimation of variation from Cont, Das.             ---
# -----------------------------------------------------------------------


def estimate_w(X, p, K):
    """
    Calculates estimated normalized p-variation from "Fact or artefact".

    Args:
        X: trajectory of the process.
        p: order of the variation to be estimated.
        K: size of subpartition.

    Returns:
        Estimation W of normalized variation.
    """
    L = len(X)
    K_grid = range(0, L, (L - 1) // K)

    # truncate X to have exactly K groups of size (L - 1) // K
    X = X[:(K_grid[-1] + 1)]
    L = len(X)
    K = (L - 1) // ((L - 1) // K)
    delta = 1 / (L - 1)

    ids = np.repeat(range(K), (L - 1) // K)[:X.size - 1]
    W = np.abs(np.diff(X[K_grid]))**p / (np.bincount(ids, np.abs(np.diff(X))**p)) * delta * ((L - 1) // K)
    return W.sum()


def w_estimator(X, K=None):
    """
    Calculates the estimation of H solving the equation W(X, 1/H, K) = 1.

    Args:
        X: trajectory of the process.
        K: size of subpartition.

    Returns:
        The w-stimation of Hurst exponent H.
    """
    if K is None:
        K = int(np.sqrt(len(X)))
    X = np.array(X)

    def func(h):
        return estimate_w(X, 1 / h, K) - 1

    # ToDo: write an adequate choice of h_min
    h_min, h_max = 0.1, 2
    while func(h_min) * func(h_max) > 0:
        h_min /= 2
    return root_scalar(func, bracket=[h_min, 2], method='bisect').root


# -----------------------------------------------------------------------
# ---             Adopted Whittle estimator by Fukasawa               ---
# -----------------------------------------------------------------------


def adopted_whittle_estimator(X, m, delta=1, x0=None):
    """
    Estimates the paramters (H, sigma) for variance process X, satisfying SDE
    d log(v_t) = kappa * dt + sigma * dW_t^H.

    Args:
        X: variance trajectory.
        m: number of points in one day (i.e. size of sample used for RV estimation).
        delta: time step (between days).
        x0: initial values of (H, sigma) in minimization.

    Returns:
        Maximum likelihood estimator of the parameters (H, sigma)
    """
    PSI = 1e-5
    K = 500
    J = 20

    Y = np.diff(np.log(X))
    n = Y.size

    def likelihood_u(H, nu):
        H = max(min(H, 0.99), 0.001)

        N_GRID = 1000
        lam_grid = np.linspace(PSI, np.pi, N_GRID)
        C = 1 / (2*np.pi) * gamma(2*H + 1) * np.sin(np.pi*H)  # page 24
        d1 = (2*np.pi*np.arange(1, K+1).reshape((-1, 1)) +
              lam_grid.reshape((1, -1)))**(-3 - 2*H) + (2*np.pi*np.arange(1, K+1).reshape((-1, 1)) - lam_grid.reshape((1, -1)))**(-3 - 2*H)
        d2 = ((2*np.pi*np.arange(K, K+2).reshape((-1, 1)) +
              lam_grid.reshape((1, -1)))**(-2 - 2*H) + (2*np.pi*np.arange(K, K+2).reshape((-1, 1)) - lam_grid.reshape((1, -1)))**(-2 - 2*H)) / (2*np.pi*(2 + 2*H))
        II = np.abs(Y @ np.exp(1j * np.arange(1, n+1).reshape((-1, 1)) *
                    lam_grid.reshape((1, -1))))**2 / (2*np.pi*n)
        g = nu**2 * C * (2*(1 - np.cos(lam_grid)))**2 * (
            np.abs(lam_grid)**(-3-2*H) + d1.sum(axis=0) +
            0.5*d2.sum(axis=0)) + 2 / (m*np.pi) * (1 - np.cos(lam_grid))  # l from page 24
        integ = np.trapz(np.log(g) + II / g, lam_grid)

        A1 = PSI*np.log(nu**2 * C) + PSI*(np.log(PSI) - 1)*(1 - 2*H) + \
            PSI**(2 + 2*H) / (nu**2 * C * m * np.pi * (2 + 2*H))

        gam = np.zeros(n)
        for i in range(n):
            gam[i] = 1 / n * Y[:n-i] @ Y[i:]
        a = np.zeros(n)
        for j in range(J + 1):
            a += 1 / (2*np.pi) * (-1)**j * np.arange(n)**(2*j) / factorial(2*j) / \
                (nu**2 * C)*(PSI**(2*j + 2*H)/(2*j + 2*H) - PSI**(1 + 2*j + 4*H) / (nu**2 * C * m * np.pi * (1 + 2*j + 4*H)))
        A2 = gam[0]*a[0] + 2*gam[1:] @ a[1:]

        u = 1 / (2*np.pi) * (integ + A1 + A2)
        return u

    if x0 is None:
        H0 = 0.5
        sigma0 = 1
    else:
        H0, sigma0 = x0

    (H_opt, nu_opt) = minimize(fun=lambda x: likelihood_u(x[0], x[1]),
                               x0=[H0, sigma0 * delta**H0],
                               method='L-BFGS-B').x
    sigma_opt = nu_opt / (delta**H_opt)
    return H_opt, sigma_opt
