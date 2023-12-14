import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def rs_regression(h, n_max=None, is_plotting=True):
    """
    Provides R/S analysis for log-increments h.

    Args:
        h: log-increments of index.
        n_max: maximal n for statistic to be calculated.
        is_plotting: whether to plot the statistic and regression.
    Returns:
        Regression coefficient H.
    """
    if n_max is None:
        n_max = h.size

    H = np.cumsum(h)
    ns = np.arange(2, n_max + 1)
    Qs = []

    for n in ns:
        H_dev = H[:n] - np.arange(1, n+1) / n * H[n-1]
        R = np.max(H_dev) - np.min(H_dev)
        S = np.std(h[:n])
        Qs.append(R / S)

    model = LinearRegression()
    model.fit(np.log(ns).reshape((-1, 1)), np.log(Qs))

    H = model.coef_[0]
    b = model.intercept_

    if is_plotting:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(ns, Qs, 'b', label='R/S statistic')
        ax.loglog(ns, np.exp(b)*ns**H, 'r--', label=f'c*n^{round(H, 2)}')
        ax.grid('on')
        ax.legend()
        ax.set_xlabel('n')
        ax.set_ylabel('R/S')
    return H
