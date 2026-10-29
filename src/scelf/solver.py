"""Exact covariance solver for the working Gaussian model."""

import numpy as np

from .numerics import _K_dot_numba, _group_sum_numba, _solve_vector_numba, _solve_matrix_numba

class ExactVSolver:
    """
    Exact solver for V = Sigma^{-1}, where

        Sigma = W^{-1} + tau Z Z^T.

    Woodbury gives

        V v = W v - W Z diag(tau / (1 + tau s_i)) Z^T W v,

    where s_i = sum_{cells in donor i} w_m.
    """

    def __init__(
        self,
        W: np.ndarray,
        tau: float,
        cell_donor: np.ndarray,
        n: int,
    ) -> None:
        self.W = np.maximum(np.asarray(W, dtype=np.float64), 1e-12)
        self.W_inv = 1.0 / self.W
        self.tau = float(max(tau, 1e-12))
        self.cell_donor = cell_donor
        self.n = int(n)
        self.M = int(W.shape[0])

        self.w_donor = _group_sum_numba(self.W, self.cell_donor, self.n)
        self.c_coef = 1.0 / np.maximum(1.0 + self.tau * self.w_donor, 1e-12)
        self.wood_coef = self.tau * self.c_coef

    def solve(self, v: np.ndarray) -> np.ndarray:
        return _solve_vector_numba(
            np.asarray(v, dtype=np.float64), self.W, self.wood_coef, self.cell_donor, self.n
        )

    def solve_matrix(self, v: np.ndarray) -> np.ndarray:
        """Apply V to multiple right-hand sides in one compiled pass."""
        return _solve_matrix_numba(
            np.asarray(v, dtype=np.float64), self.W, self.wood_coef, self.cell_donor, self.n
        )

    def dotK(self, v: np.ndarray) -> np.ndarray:
        return _K_dot_numba(np.asarray(v, dtype=np.float64), self.cell_donor, self.n)

    def logdet_sigma(self) -> float:
        return (
            -float(np.sum(np.log(self.W)))
            + float(np.sum(np.log1p(self.tau * self.w_donor)))
        )

