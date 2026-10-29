"""Numerical kernels used by scELF."""

import numpy as np
from numba import njit


@njit(cache=True)
def _weighted_group_sum_numba(values, weights, cell_donor, n):
    """Z.T @ (W * values), without a cell-sized product allocation."""
    out = np.zeros(n, dtype=np.float64)
    for k in range(values.shape[0]):
        out[cell_donor[k]] += weights[k] * values[k]
    return out


@njit(cache=True)
def _group_sum_matrix_numba(values, cell_donor, n):
    out = np.zeros((n, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[0]):
        for j in range(values.shape[1]):
            out[cell_donor[k], j] += values[k, j]
    return out


@njit(cache=True)
def _weighted_group_sum_matrix_numba(values, weights, cell_donor, n):
    out = np.zeros((n, values.shape[1]), dtype=np.float64)
    for k in range(values.shape[0]):
        for j in range(values.shape[1]):
            out[cell_donor[k], j] += weights[k] * values[k, j]
    return out


@njit(cache=True)
def _solve_vector_numba(v, weights, wood_coef, cell_donor, n):
    d = _weighted_group_sum_numba(v, weights, cell_donor, n)
    out = np.empty(v.shape[0], dtype=np.float64)
    for k in range(v.shape[0]):
        i = cell_donor[k]
        out[k] = weights[k] * v[k] - weights[k] * (wood_coef[i] * d[i])
    return out


@njit(cache=True)
def _solve_matrix_numba(v, weights, wood_coef, cell_donor, n):
    d = _weighted_group_sum_matrix_numba(v, weights, cell_donor, n)
    out = np.empty(v.shape, dtype=np.float64)
    for k in range(v.shape[0]):
        i = cell_donor[k]
        for j in range(v.shape[1]):
            out[k, j] = weights[k] * v[k, j] - weights[k] * (wood_coef[i] * d[i, j])
    return out

# ============================================================
# Numba helpers
# ============================================================

@njit(cache=True)
def _group_sum_numba(values: np.ndarray, cell_donor: np.ndarray, n: int) -> np.ndarray:
    s = np.zeros(n, dtype=np.float64)
    M = values.shape[0]
    for k in range(M):
        i = int(cell_donor[k])
        s[i] += float(values[k])
    return s


@njit(cache=True)
def _expand_to_cells_numba(donor_vec: np.ndarray, cell_donor: np.ndarray, M: int) -> np.ndarray:
    out = np.empty(M, dtype=np.float64)
    for k in range(M):
        out[k] = float(donor_vec[int(cell_donor[k])])
    return out


@njit(cache=True)
def _K_dot_numba(v: np.ndarray, cell_donor: np.ndarray, n: int) -> np.ndarray:
    s = _group_sum_numba(v, cell_donor, n)
    return _expand_to_cells_numba(s, cell_donor, v.shape[0])


@njit(cache=True)
def _compute_s_vec_numba(G: np.ndarray, u_coef: np.ndarray) -> np.ndarray:
    """
    Compute s_j = sum_i G_ij^2 * u_coef_i.
    """
    n, p = G.shape
    s = np.zeros(p, dtype=np.float64)
    for j in range(p):
        acc = 0.0
        for i in range(n):
            gij = G[i, j]
            acc += gij * gij * u_coef[i]
        s[j] = acc
    return s


@njit(cache=True)
def _Gt_times_vec_numba(G: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute G^T v.
    """
    n, p = G.shape
    out = np.zeros(p, dtype=np.float64)
    for j in range(p):
        acc = 0.0
        for i in range(n):
            acc += G[i, j] * v[i]
        out[j] = acc
    return out

