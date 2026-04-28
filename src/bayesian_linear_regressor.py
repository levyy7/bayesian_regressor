import math

import numpy as np


class BayesianLinearRegressor:

    def __init__(self, sigma2: float = 1.0, sigma2_v: float = 1.0) -> None:
        """
        Constructor for Bayesian Linear Regression.

        Args:
            sigma2: The variance of the target distribution (noise).
            sigma2_v: The prior variance of the weights. Acts as regularization parameter.
        """
        self.sigma2: float = sigma2
        self.sigma2_v: float = sigma2_v

        self.mean_post = None
        self.cov_post = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim != 1:
            raise ValueError("y must be 1D, shape (n,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.mean_post, self.cov_post = self._compute_gaussian_posterior_parameters(
            X, y, self.sigma2, self.sigma2_v
        )

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._compute_gaussian_prediction_parameters(X_test, self.sigma2, self.mean_post, self.cov_post)

    @staticmethod
    def _compute_gaussian_posterior_parameters(
            X: np.ndarray, y: np.ndarray, sigma2: float, sigma2_v: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the Gaussian posterior over weights given observations.

        Σ_n = ( (1/σ_v²)·I  +  XᵀX/σ² )⁻¹
        μ_n = (1/σ²) · Σ_n · Xᵀy

        Args:
            X:       Design matrix of shape (n, d).
            y:       Target vector of shape (n,).
            sigma2:  Noise variance σ² of the likelihood.
            sigma2_v: Prior weight variance σ_v².

        Returns:
            A tuple (mean_post, cov_post) where:
                mean_post:    Posterior mean of shape (d,).
                cov_post: Posterior covariance of shape (d, d).
        """
        cov_post = np.linalg.inv((1.0 / sigma2_v) * np.eye(X.shape[1]) + X.T @ X / sigma2)
        mean_post = (1.0 / sigma2) * cov_post @ X.T @ y

        return mean_post, cov_post

    @staticmethod
    def _compute_gaussian_prediction_parameters(
            X_test: np.ndarray, sigma2: float, mean_post: np.ndarray, cov_post: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_pred = X_test @ mean_post

        A = X_test @ cov_post
        var_pred = np.sum(A * X_test, axis=1) + sigma2

        return mean_pred, var_pred