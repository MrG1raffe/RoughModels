import numpy as np
import pandas as pd
from scipy.special import hyp2f1
from numpy.linalg import cholesky

class rBergomi():
    def __init__(self, H, T=4, rho=0):
        self.H = H
        self.gamma = 0.5 - self.H
        self.T = T
        self.rho = rho
        self.time = None

    def G(self, x):
        gamma = self.gamma
        return (1 - 2*gamma) * x ** (-gamma) * hyp2f1(1, gamma, 2 - gamma, 1/x) / (1 - gamma)

    def covariance_matrix(self, m: int = 10, n: int = 10):
        '''
        m - int, number of time steps
        n - int, number of simulations
        '''
        epsilon = 1e-10
        H = self.H
        Dh = np.sqrt(2*self.H) / (H + 0.5)
        covariance = np.eye(2 * m)
        time = np.linspace(0, self.T, m) + epsilon
        self.time = time
        time_line, time_raw = np.meshgrid(time, time)
        minima = np.min(np.stack((time_line, time_raw)), axis=0)
        maxima = np.max(np.stack((time_line, time_raw)), axis=0)
        covariance[:m, :m] = minima
        WZ_cov = self.rho * Dh * (time_line ** (H+0.5) - (time_line - minima) ** (H + 0.5))
        covariance[m:, :m] = WZ_cov
        covariance[:m, m:] = WZ_cov.T
        covariance[m:, m:] = minima ** (2*H) * self.G(maxima/minima)
        return covariance

    def WZ_sample(self, m: int = 20, n: int = 20):
        epsilon = 1e-10
        covariance = self.covariance_matrix(m, n)
        A = cholesky(covariance+epsilon)
        samples = np.random.randn(2*m, n)
        samples = A @ samples
        return samples[:m], samples[m:]

    def sample_v(self, W, eta, xi):
        gamma = self.gamma
        time = self.time
        v = np.exp(eta * W - eta * eta / 2 * np.expand_dims(time, axis=-1) ** (1 - 2 * gamma))
        return xi * v

    def sample_S(self, v, Z):
        time = self.time
        dt = time[1] - time[0]
        dZ = np.diff(Z, axis=0)
        lnS = np.zeros(Z.shape)
        lnS[0, :] = 0
        for i in range(1, v.shape[0]):
            lnS[i, :] = lnS[i-1, :] + np.sqrt(v[i-1, :]) * dZ[i-1, :] - 0.5 * v[i-1, :] * v[i-1, :] * dt
        return np.exp(lnS)
