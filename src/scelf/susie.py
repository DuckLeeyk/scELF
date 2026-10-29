"""
SuSiE model implementation using NumPy and SciPy.

This module implements a NumPy/Scipy-based version of the SuSiE (Sum of Single Effects) model
for sparse regression and fine-mapping. It includes utilities for column statistics, correlation,
credible set construction, posterior inclusion probabilities (PIPs), and auxiliary univariate regression helpers.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from typing import Optional, Dict, Any, List, Tuple, Union
from .backend import resolve_backend, _load_cpp


def compute_Xb(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the matrix-vector product X @ b.

    Parameters
    - X: Design matrix of shape (n, p).
    - b: Vector of coefficients of length p.

    Returns
    - Xb: Vector of length n equal to X @ b.
    """
    return X @ b

def compute_MXt(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    # Returns an array of shape (L, n); the l-th row equals X @ M[l,]^T
    """
    Compute (X @ M.T).T in a memory-friendly manner.

    Parameters
    - M: Matrix of shape (L, p); each row is a coefficient vector.
    - X: Design matrix of shape (n, p).

    Returns
    - Y: Matrix of shape (L, n) where Y[l, :] = X @ M[l,]^T.
    """
    return (X @ M.T).T

def compute_Xty(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute X.T @ y.

    Parameters
    - X: Design matrix of shape (n, p).
    - y: Response vector of length n.

    Returns
    - Xty: Vector of length p equal to X.T @ y.
    """
    return X.T @ y

def compute_colSds(X: np.ndarray) -> np.ndarray:
    """
    Compute column-wise standard deviations using the unbiased variance (ddof=1).

    Notes
    - For numerical stability, negative variances due to rounding are clamped to 0.
    - Columns with zero variance return sd = 0.

    Parameters
    - X: Array of shape (n, p).

    Returns
    - sds: Vector of length p containing column standard deviations.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    col_mean = np.mean(X, axis=0)
    col_mean_sq = np.mean(X**2, axis=0)
    # Unbiased variance (ddof=1) to match standard statistical definition
    var = (col_mean_sq - col_mean**2) * (n / max(n - 1, 1))
    var = np.maximum(var, 0.0)
    return np.sqrt(var)

def compute_colstats(X: np.ndarray, center: bool = True, scale: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute column statistics needed for standardized linear algebra operations.

    Definitions
    - cm: Column means (or zeros if center=False).
    - csd: Column standard deviations; zeros are set to 1 to avoid division by zero if scale=True.
    - d: Sum of squares of standardized columns, i.e., colSums(((X - cm)/csd)^2),
         computed in a numerically stable way.

    Parameters
    - X: Array of shape (n, p).
    - center: Whether to subtract column means.
    - scale: Whether to scale by column standard deviations.

    Returns
    - A dict with keys:
      - "cm": (p,) array of column means (or zeros).
      - "csd": (p,) array of column SDs (or ones).
      - "d": (p,) array of squared norms after centering/scaling.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    cm = np.mean(X, axis=0) if center else np.zeros(p, float)
    if scale:
        csd = compute_colSds(X)
        csd[csd == 0] = 1.0  # Avoid division by zero in scaling
    else:
        csd = np.ones(p, float)

    # d = colSums(((X - cm)/csd)^2) computed in a numerically stable way
    col_mean = np.mean(X, axis=0)
    d_raw = n * (col_mean**2) + (n - 1) * (compute_colSds(X)**2)
    d = (d_raw - n * (cm**2)) / (csd**2)
    return dict(cm=cm, csd=csd, d=d)

def muffled_corr(X: np.ndarray) -> np.ndarray:
    """
    Compute a numerically safe correlation matrix.

    Behavior
    - Centers columns using sample means.
    - Scales by sample SD (ddof=1); for zero-variance columns, uses sd=1 to avoid NaN.
    - Sets rows/cols corresponding to zero-variance columns to 0 and diagonal to 1.

    Parameters
    - X: Array of shape (n, p).

    Returns
    - R: (p, p) correlation matrix with safe handling for degenerate columns.
    """
    X = np.asarray(X, float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1)
    zero = sd < 1e-15
    sd_safe = sd.copy()
    sd_safe[zero] = 1.0
    Xn = Xc / sd_safe
    R = (Xn.T @ Xn) / (n - 1)
    R[zero, :] = 0
    R[:, zero] = 0
    np.fill_diagonal(R, 1.0)
    return R

def is_symmetric_matrix(X: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Check if matrix X is symmetric within a given absolute tolerance.

    Parameters
    - X: Square matrix.
    - tol: Absolute tolerance.

    Returns
    - True if X is symmetric; False otherwise.
    """
    return np.allclose(X, X.T, atol=tol, rtol=0)

def n_in_CS_x(x: np.ndarray, coverage: float = 0.9) -> int:
    """
    Compute how many top entries are needed to reach the target coverage.

    Parameters
    - x: Probability weights (e.g., a row of alpha) of length p.
    - coverage: Target coverage in (0, 1].

    Returns
    - n: Minimal number of largest entries summing to at least coverage.
    """
    xs = np.sort(x)[::-1]
    csum = np.cumsum(xs)
    return int(np.sum(csum < coverage) + 1)

def in_CS_x(x: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    """
    Indicator vector of membership in the credible set for a single effect.

    Parameters
    - x: Probability weights (length p).
    - coverage: Target coverage.

    Returns
    - indicator: 0/1 array of length p; 1 indicates inclusion in CS.
    """
    n = n_in_CS_x(x, coverage)
    o = np.argsort(x)[::-1]
    result = np.zeros_like(x, dtype=int)
    result[o[:n]] = 1
    return result

def in_CS(res: Union[Dict[str, Any], np.ndarray], coverage: float = 0.9) -> np.ndarray:
    """
    Compute credible set membership indicators for all L effects.

    Parameters
    - res: Either a result dict containing "alpha" or the alpha array (L, p).
    - coverage: Target coverage level.

    Returns
    - status: (L, p) 0/1 matrix indicating CS membership per effect.
    """
    if isinstance(res, dict):
        alpha = res["alpha"]
    else:
        alpha = np.asarray(res)
    status = np.vstack([in_CS_x(alpha[l, :], coverage) for l in range(alpha.shape[0])])
    return status

def get_purity(pos: List[int],
               X: Optional[np.ndarray],
               Xcorr: Optional[np.ndarray],
               squared: bool = False,
               n: int = 100,
               use_rfast: Optional[bool] = None,
               rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """
    Compute purity metrics (min/mean/median absolute correlation) within a set of variables.

    Parameters
    - pos: Indices of variables in the set.
    - X: Design matrix (n, p) if Xcorr is not provided.
    - Xcorr: Precomputed correlation matrix (p, p). Use either X or Xcorr, not both.
    - squared: If True, use squared correlations; otherwise absolute correlations.
    - n: If |pos| > n, subsample n indices from pos to cap computation.
    - use_rfast: Placeholder for API compatibility (unused).
    - rng: Optional random generator for subsampling.

    Returns
    - (min_abs, mean_abs, median_abs): Purity summary statistics.
    """
    pos = list(pos)
    if len(pos) == 1:
        return (1.0, 1.0, 1.0)
    if len(pos) > n:
        if rng is None:
            rng = np.random.default_rng(0)
        pos = list(rng.choice(pos, size=n, replace=False))
    if Xcorr is None:
        X_sub = np.asarray(X[:, pos])
        R = muffled_corr(X_sub)
    else:
        R = np.asarray(Xcorr)[np.ix_(pos, pos)]
        R = 0.5 * (R + R.T)  # Force symmetry
    # Extract upper-triangular (pairwise) correlations
    idx = np.triu_indices(len(pos), k=1)
    vals = np.abs(R[idx])
    if squared:
        vals = vals**2
    if vals.size == 0:
        return (1.0, 1.0, 1.0)
    return (float(np.min(vals)),
            float(np.mean(vals)),
            float(np.median(vals)))

def susie_get_cs_py(res: Dict[str, Any],
                    X: Optional[np.ndarray] = None,
                    Xcorr: Optional[np.ndarray] = None,
                    coverage: float = 0.95,
                    min_abs_corr: float = 0.5,
                    dedup: bool = True,
                    squared: bool = False,
                    check_symmetric: bool = True,
                    n_purity: int = 100,
                    use_rfast: Optional[bool] = None) -> Dict[str, Any]:
    """
    Construct credible sets (CS) from SuSiE results and evaluate set purity.

    Workflow
    - Determine per-effect CS based on alpha and target coverage.
    - Optionally deduplicate identical sets across effects.
    - If X or Xcorr is provided, compute set purity and filter by min_abs_corr.
    - Return CS indices, purity stats, coverage claims, and requested coverage.

    Parameters
    - res: SuSiE-like result dict containing "alpha", optionally "V" and "null_index".
    - X: Design matrix (n, p). Provide X or Xcorr, not both.
    - Xcorr: Correlation matrix (p, p).
    - coverage: Target coverage for CS construction.
    - min_abs_corr: Minimum pairwise absolute correlation threshold for purity filtering.
    - dedup: Remove duplicate sets across effects.
    - squared: If True, purity uses squared correlations.
    - check_symmetric: If True, enforce symmetry of Xcorr.
    - n_purity: Subsampling cap for large sets when computing purity.
    - use_rfast: Placeholder for API compatibility (unused).

    Returns
    - A dict with keys:
      - "cs": Mapping from "Lk" to list of indices (or None if no sets).
      - "purity": Dict of purity summaries per set (or None if X/Xcorr not given).
      - "cs_index": List of indices of included effects.
      - "coverage": Declared coverage per set.
      - "requested_coverage": The coverage requested by the user.
    """
    if (X is not None) and (Xcorr is not None):
        raise ValueError("Only one of X or Xcorr should be specified")
    if (Xcorr is not None) and check_symmetric and (not is_symmetric_matrix(Xcorr)):
        Xcorr = 0.5 * (Xcorr + Xcorr.T)

    null_index = res.get("null_index", 0) or 0
    alpha = res["alpha"]

    V = res.get("V", None)
    if isinstance(V, np.ndarray):
        include_idx = (V > 1e-9)
    elif isinstance(V, (float, int)):
        include_idx = np.ones(alpha.shape[0], dtype=bool)
    else:
        include_idx = np.ones(alpha.shape[0], dtype=bool)

    status = in_CS(alpha, coverage)
    cs = []
    claimed_coverage = []
    for i in range(status.shape[0]):
        pos = np.where(status[i, :] != 0)[0].tolist()
        cs.append(pos)
        claimed_coverage.append(float(np.sum(alpha[i, pos])))

    include_idx = include_idx & (np.array([len(v) for v in cs]) > 0)

    if dedup:
        # Track unique sets; keep the first occurrence
        seen = set()
        mask = []
        for i, sset in enumerate(cs):
            key = tuple(sorted(sset))
            if (key not in seen) and include_idx[i]:
                seen.add(key)
                mask.append(True)
            else:
                mask.append(False)
        include_idx = include_idx & np.array(mask)

    if np.sum(include_idx) == 0:
        return dict(cs=None, coverage=None, requested_coverage=coverage)

    cs = [cs[i] for i in np.where(include_idx)[0]]
    claimed_coverage = [claimed_coverage[i] for i in np.where(include_idx)[0]]

    if (Xcorr is None) and (X is None):
        # Return CS without purity if neither X nor Xcorr is supplied
        cs_dict = {f"L{k}": [int(idx) for idx in sset] for k, sset in enumerate(cs)}
        coverage_map = {str(k): float(cov) for k, cov in enumerate(claimed_coverage)}
        return dict(
            cs=cs_dict,
            purity=None,
            cs_index=list(range(len(cs))),
            coverage=coverage_map,
            requested_coverage=float(coverage)
        )

    # Compute purity with optional subsampling for large sets
    purity = []
    rng = np.random.default_rng(0)
    for sset in cs:
        if (null_index > 0) and (null_index in sset):
            purity.append((-9.0, -9.0, -9.0))  # Mark sets containing the null variable
        else:
            purity.append(get_purity(sset, X, Xcorr, squared=squared, n=n_purity, rng=rng))
    purity = np.asarray(purity, float)

    threshold = (min_abs_corr**2) if squared else min_abs_corr
    is_pure = np.where(purity[:, 0] >= threshold)[0]
    if is_pure.size == 0:
        return dict(cs=None, coverage=None, requested_coverage=coverage)

    cs = [cs[i] for i in is_pure]
    purity = purity[is_pure, :]
    cs_index = list(range(len(cs)))
    covg = np.asarray(claimed_coverage, float)[is_pure]

    # Order sets by minimum absolute correlation descending
    ordering = np.argsort(purity[:, 0])[::-1]
    cs = [cs[i] for i in ordering]
    purity = purity[ordering, :]
    cs_index = [int(cs_index[i]) for i in ordering]
    covg = covg[ordering]

    cs_dict = {f"L{k}": [int(idx) for idx in sset] for k, sset in enumerate(cs)}
    purity_dict = {
        "min_abs_corr": {f"L{k}": float(purity[k, 0]) for k in range(len(cs))},
        "mean_abs_corr": {f"L{k}": float(purity[k, 1]) for k in range(len(cs))},
        "median_abs_corr": {f"L{k}": float(purity[k, 2]) for k in range(len(cs))}
    }
    coverage_map = {str(k): float(covg[k]) for k in range(len(cs))}

    return dict(
        cs=cs_dict,
        purity=purity_dict,
        cs_index=[int(i) for i in cs_index],
        coverage=coverage_map,
        requested_coverage=float(coverage)
    )

def susie_get_pip_py(res: Union[Dict[str, Any], np.ndarray], prune_by_cs: bool = False, prior_tol: float = 1e-9) -> np.ndarray:
    """
    Compute posterior inclusion probabilities (PIPs).

    Parameters
    - res: Either a SuSiE-like dict containing "alpha" (and optionally "V", "null_index", "sets")
           or an alpha array (L, p).
    - prune_by_cs: If True and res contains credible set indices, restrict to those effects.
    - prior_tol: Threshold to determine which effects are retained based on prior variance V.

    Returns
    - pip: Vector of length p with PIPs.
    """
    if isinstance(res, dict):
        alpha = res["alpha"]
        null_index = res.get("null_index", None)
        if null_index is not None and null_index > 0:
            alpha = np.delete(alpha, null_index, axis=1)
        V = res.get("V", None)
        if isinstance(V, np.ndarray) or isinstance(V, (float, int)):
            if np.ndim(V) == 0:
                include_idx = np.arange(alpha.shape[0])
            else:
                include_idx = np.where(np.asarray(V) > prior_tol)[0]
        else:
            include_idx = np.arange(alpha.shape[0])
        if prune_by_cs and res.get("sets", None) is not None and ("cs_index" in res["sets"]):
            include_idx = np.intersect1d(include_idx, np.asarray(res["sets"]["cs_index"], dtype=int))
        alpha_use = alpha[include_idx, :] if include_idx.size > 0 else np.zeros((1, alpha.shape[1]))
    else:
        alpha_use = np.asarray(res)
    alpha_use = np.nan_to_num(alpha_use, nan=0.0, posinf=1.0, neginf=0.0)
    return 1.0 - np.prod(1.0 - alpha_use, axis=0)


# ==============================
# Initialization
# ==============================

def init_setup(n: int, p: int, L: int, scaled_prior_variance, residual_variance,
               prior_weights, null_weight, varY: float, standardize: bool) -> Dict[str, Any]:
    """
    Initialize core SuSiE state with default values and checks.

    Parameters
    - n: Number of samples.
    - p: Number of predictors (possibly including a null column).
    - L: Number of single-effect components.
    - scaled_prior_variance: Scalar or array controlling prior variance scaling.
    - residual_variance: Initial residual variance; if None, set to varY.
    - prior_weights: Prior over variables; if None, uniform over p variables.
    - null_weight: If not None, indicates presence/weight of a null variable.
    - varY: Variance of y (centered).
    - standardize: If True, requires scaled_prior_variance <= 1.

    Returns
    - s: Dict with initialized arrays and hyperparameters for SuSiE.
    """
    spv = np.asarray(scaled_prior_variance, float)
    if np.any(spv < 0):
        raise ValueError("Scaled prior variance should be positive")
    if np.any(spv > 1) and standardize:
        raise ValueError("Scaled prior variance should be <= 1 when standardize is True")
    if residual_variance is None:
        residual_variance = varY
    if prior_weights is None:
        prior_weights = np.repeat(1.0 / p, p)
    else:
        prior_weights = np.asarray(prior_weights, float)
        if np.all(prior_weights == 0):
            raise ValueError("Prior weight must be > 0 for at least one variable.")
        prior_weights = prior_weights / np.sum(prior_weights)
    if len(prior_weights) != p:
        raise ValueError("Prior weights must have length p")
    if p < L:
        L = p
    s = dict(
        alpha=np.full((L, p), 1.0 / p),
        mu=np.zeros((L, p)),
        mu2=np.zeros((L, p)),
        Xr=np.zeros(n),
        KL=np.full(L, np.nan),
        lbf=np.full(L, np.nan),
        lbf_variable=np.full((L, p), np.nan),
        sigma2=float(residual_variance),
        V=spv * float(varY),
        pi=prior_weights
    )
    if null_weight is None:
        s["null_index"] = 0
    else:
        s["null_index"] = p - 1
    return s

def init_finalize(s: Dict[str, Any], X: Optional[np.ndarray] = None, Xr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Finalize SuSiE state: broadcast scalars, validate dimensions, and compute Xr if needed.

    Parameters
    - s: State dict from init_setup (or compatible).
    - X: Design matrix (n, p); if provided, compute Xr from current posterior means.
    - Xr: If provided, overrides s["Xr"].

    Returns
    - s: Validated and finalized state dict ready for iteration.
    """
    s = dict(s)
    if np.ndim(s["V"]) == 0:
        s["V"] = np.repeat(float(s["V"]), s["alpha"].shape[0])
    s["sigma2"] = float(s["sigma2"])
    if s["sigma2"] <= 0:
        raise ValueError("Residual variance sigma2 must be positive")
    if not np.all(np.asarray(s["V"]) >= 0):
        raise ValueError("Prior variance must be non-negative")
    if s["mu"].shape != s["mu2"].shape or s["mu"].shape != s["alpha"].shape:
        raise ValueError("Dimension mismatch among mu, mu2 and alpha")
    if s["alpha"].shape[0] != len(s["V"]):
        raise ValueError("Length of V must equal nrow(alpha)")
    if Xr is not None:
        s["Xr"] = Xr
    if X is not None:
        # Compute fitted values from current posterior means and responsibilities
        s["Xr"] = compute_Xb(X, np.sum(s["mu"] * s["alpha"], axis=0))
    s["KL"] = np.full(s["alpha"].shape[0], np.nan)
    s["lbf"] = np.full(s["alpha"].shape[0], np.nan)
    return s


# ==============================
# ELBO and objectives
# ==============================

def get_ER2(X: np.ndarray, Y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    """
    Compute E[||Y - Xb||^2] under the current variational posterior.

    Components
    - s["Xr"] stores X @ E[b].
    - Xr_L captures per-effect contributions X @ (alpha_l * mu_l).
    - d_attr contains column-wise squared norms for standardized design.

    Parameters
    - X: Design matrix (n, p).
    - Y: Centered response (n,).
    - s: SuSiE state dict.
    - d_attr: Precomputed column squared norms ("d" from compute_colstats).

    Returns
    - Expected residual sum of squares as a float.
    """
    Xr_L = compute_MXt(s["alpha"] * s["mu"], X)
    postb2 = s["alpha"] * s["mu2"]
    return float(np.sum((Y - s["Xr"])**2) - np.sum(Xr_L**2) + np.sum(d_attr * np.sum(postb2, axis=0)))

def SER_posterior_e_loglik(X: np.ndarray, Y: np.ndarray, s2: float, Eb: np.ndarray, Eb2: np.ndarray,
                           d_attr: np.ndarray) -> float:
    """
    Compute E[log-likelihood] for the single-effect regression posterior.

    Parameters
    - X: Design matrix (n, p).
    - Y: Residual vector (n,) for the current effect.
    - s2: Residual variance.
    - Eb: Posterior mean of b elementwise-weighted by alpha (length p).
    - Eb2: Posterior second moment of b weighted by alpha (length p).
    - d_attr: Column squared norms.

    Returns
    - Expected log-likelihood contribution for one effect.
    """
    n = X.shape[0]
    term = -0.5 * n * np.log(2 * np.pi * s2) - 0.5 / s2 * (np.sum(Y * Y)
                                                           - 2 * np.sum(Y * compute_Xb(X, Eb))
                                                           + np.sum(d_attr * Eb2))
    return float(term)

def Eloglik(X: np.ndarray, Y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    """
    Compute expected log-likelihood for the full model.

    Parameters
    - X: Design matrix (n, p).
    - Y: Centered response (n,).
    - s: SuSiE state.
    - d_attr: Column squared norms.

    Returns
    - E[log p(Y | b, sigma2)] under the variational posterior.
    """
    n = X.shape[0]
    return float(-(n / 2) * np.log(2 * np.pi * s["sigma2"]) - (1.0 / (2 * s["sigma2"])) * get_ER2(X, Y, s, d_attr=d_attr))

def get_objective(X: np.ndarray, Y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    """
    Compute the variational objective (ELBO): E[loglik] - sum(KL).

    Parameters
    - X, Y: Data.
    - s: SuSiE state containing KL terms.
    - d_attr: Column squared norms.

    Returns
    - Scalar ELBO value.
    """
    return float(Eloglik(X, Y, s, d_attr=d_attr) - np.nansum(s["KL"]))


def _logsumexp_full(lpo: np.ndarray) -> float:
    """
    Numerically stable logsumexp for a 1D array.

    Parameters
    - lpo: Log-probabilities or scores.

    Returns
    - log(sum(exp(lpo))).
    """
    m = np.max(lpo)
    return float(m + np.log(np.sum(np.exp(lpo - m))))

def _loglik_core(Vv: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """
    Core marginal log-likelihood for a single-effect model across all variables.

    Notes
    - Invalid columns (non-finite or non-positive shat2) contribute Bayes factor = 1.

    Parameters
    - Vv: Prior variance for effect.
    - betahat: Per-variable effect estimates.
    - shat2: Per-variable variance estimates.
    - prior: Prior weights over variables (sum to 1).

    Returns
    - Scalar log-likelihood.
    """
    # Invalid columns contribute BF=1 (lbf=0), so lpo = log(prior)
    lpo = np.log(prior + np.sqrt(np.finfo(float).eps))
    mask = np.isfinite(shat2) & (shat2 > 0)
    if np.any(mask):
        denom = Vv + shat2[mask]
        denom = np.maximum(denom, np.finfo(float).tiny)
        lbf_mask = ((-0.5 * np.log(2 * np.pi * denom) - 0.5 * (betahat[mask]**2) / denom) -
                    (-0.5 * np.log(2 * np.pi * shat2[mask]) - 0.5 * (betahat[mask]**2) / shat2[mask]))
        lpo[mask] = lpo[mask] + lbf_mask
    return _logsumexp_full(lpo)

def neg_loglik_logscale(lV: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """
    Negative log-likelihood with V parameterized on the log scale.

    Parameters
    - lV: log(V).
    - betahat, shat2, prior: As in _loglik_core.

    Returns
    - Negative log-likelihood.
    """
    return float(-_loglik_core(np.exp(lV), betahat, shat2, prior))

def loglik_grad(V: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """
    Gradient of the log-likelihood with respect to V (not log V).

    Notes
    - Uses softmax over variables with invalid columns contributing prior only.

    Parameters
    - V: Prior variance (positive).
    - betahat, shat2, prior: As in _loglik_core.

    Returns
    - Scalar gradient d/dV log-likelihood.
    """
    # Alpha is softmax over all columns (invalid columns contribute prior only)
    p = prior.size
    lpo = np.log(prior + np.sqrt(np.finfo(float).eps))
    mask = np.isfinite(shat2) & (shat2 > 0)
    grad_sum = 0.0
    if np.any(mask):
        denom = V + shat2[mask]
        denom = np.maximum(denom, np.finfo(float).tiny)
        lbf_mask = ((-0.5 * np.log(2 * np.pi * denom) - 0.5 * (betahat[mask]**2) / denom) -
                    (-0.5 * np.log(2 * np.pi * shat2[mask]) - 0.5 * (betahat[mask]**2) / shat2[mask]))
        lpo[mask] = lpo[mask] + lbf_mask

        m = np.max(lpo)
        w = np.exp(lpo - m)
        sw = np.sum(w)
        alpha_full = w / sw if sw > 0 else np.repeat(1.0 / p, p)
        T2 = (betahat[mask]**2) / shat2[mask]
        grad_vec = 0.5 * (1.0 / denom) * ((shat2[mask] / denom) * T2 - 1.0)
        grad_vec[np.isnan(grad_vec)] = 0.0
        grad_sum = float(np.sum(alpha_full[mask] * grad_vec))
    else:
        grad_sum = 0.0
    return grad_sum

def negloglik_grad_logscale(lV: float, betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """
    Negative gradient of log-likelihood with respect to log V.

    Uses chain rule: d/d(log V) = V * d/dV.

    Parameters
    - lV: log(V).
    - betahat, shat2, prior: As in _loglik_core.

    Returns
    - Negative gradient value.
    """
    V = np.exp(lV)
    return float(-V * loglik_grad(V, betahat, shat2, prior))

def est_V_uniroot(betahat: np.ndarray, shat2: np.ndarray, prior: np.ndarray) -> float:
    """
    Estimate V by finding a root of the log V-gradient using Brent's method.

    Parameters
    - betahat, shat2, prior: As in _loglik_core.

    Returns
    - Estimated prior variance V (> 0), or a fallback value if root-finding fails.
    """
    # Find root of gradient in log(V) using brentq with bracket expansion
    def g(lV):
        return negloglik_grad_logscale(lV, betahat, shat2, prior)

    bounds = [(-10.0, 10.0), (-20.0, 20.0), (-30.0, 30.0)]
    for a, b in bounds:
        fa, fb = g(a), g(b)
        if np.isfinite(fa) and np.isfinite(fb) and (fa * fb <= 0):
            sol = root_scalar(g, bracket=(a, b), method="brentq", xtol=1e-8, rtol=1e-8, maxiter=200)
            if sol.converged:
                return float(np.exp(sol.root))
    return float(np.exp(0.5 * (bounds[-1][0] + bounds[-1][1])))

def optimize_prior_variance(optimize_V: str,
                            betahat: np.ndarray,
                            shat2: np.ndarray,
                            prior: np.ndarray,
                            alpha: Optional[np.ndarray] = None,
                            post_mean2: Optional[np.ndarray] = None,
                            V_init: Optional[float] = None,
                            check_null_threshold: float = 0.0,
                            backend: str = "numba") -> float:
    """
    Optimize or update the prior variance V according to the specified method.

    Methods
    - "optim": 1D Brent optimization on log V.
    - "uniroot": Root-finding on gradient w.r.t. log V.
    - "EM": Closed-form EM update using alpha and post_mean2.
    - "none": Keep V as is.

    Additionally checks whether V=0 (null model) is favored within a tolerance.

    Parameters
    - betahat, shat2, prior: As in _loglik_core.
    - alpha, post_mean2: Required if optimize_V == "EM".
    - V_init: Initial V.
    - check_null_threshold: If null model's log-lik + threshold >= current, set V=0.

    Returns
    - Updated V (>= 0).
    """
    V = float(V_init) if V_init is not None else 0.0
    backend, _ = resolve_backend(backend)
    likelihood = (
        _load_cpp().PriorLikelihood(
            np.ascontiguousarray(betahat, dtype=np.float64),
            np.ascontiguousarray(shat2, dtype=np.float64), np.ascontiguousarray(prior, dtype=np.float64)
        ).value if backend == "cpp" else lambda value: _loglik_core(value, betahat, shat2, prior)
    )
    if optimize_V != "simple":
        if optimize_V == "optim":
            def f(lV):
                return -likelihood(float(np.exp(lV)))
            res = minimize_scalar(f, bracket=(-30.0, 15.0), method="brent", options={"maxiter": 500})
            lV_new = float(res.x)
            if V_init is not None and np.isfinite(V_init) and (V_init > 0):
                if f(lV_new) > f(np.log(V_init)):
                    lV_new = np.log(V_init)
            V = float(np.exp(lV_new))

        elif optimize_V == "uniroot":
            V = est_V_uniroot(betahat, shat2, prior)

        elif optimize_V == "EM":
            if alpha is None or post_mean2 is None:
                raise ValueError("EM requires alpha and post_mean2")
            V = float(np.sum(alpha * post_mean2))

        elif optimize_V == "none":
            V = float(V)

        else:
            raise ValueError("Invalid option for optimize_V method")

    # If null (V=0) is competitive within threshold, prefer V=0
    if likelihood(0.0) + check_null_threshold >= likelihood(V):
        V = 0.0
    return V

def single_effect_regression(y: np.ndarray,
                             X: np.ndarray,
                             V: float,
                             residual_variance: float = 1.0,
                             prior_weights: Optional[np.ndarray] = None,
                             optimize_V: str = "none",
                             check_null_threshold: float = 0.0,
                             d_attr: Optional[np.ndarray] = None,
                             backend: str = "numba") -> Dict[str, Any]:
    """
    Perform single-effect regression (SER) update for one component.

    Parameters
    - y: Current residual vector (n,).
    - X: Design matrix (n, p).
    - V: Current prior variance for this effect.
    - residual_variance: Sigma^2.
    - prior_weights: Prior over variables (length p).
    - optimize_V: One of {"none", "optim", "uniroot", "EM"}.
    - check_null_threshold: Threshold for preferring V=0.
    - d_attr: Column squared norms (length p), required.

    Returns
    - Dict containing:
      - "alpha": Posterior responsibilities over variables (p,).
      - "mu": Posterior mean per variable (p,).
      - "mu2": Posterior second moment per variable (p,).
      - "lbf": Per-variable log Bayes factors (p,).
      - "lbf_model": Log-model evidence (scalar).
      - "V": Updated V (scalar).
      - "loglik": Log-likelihood under sigma^2 (scalar).
    """
    n, p = X.shape
    if d_attr is None:
        raise ValueError("single_effect_regression requires d_attr")

    Xty = compute_Xty(X, y)
    d = np.asarray(d_attr, float)

    # Compute per-variable OLS estimates and their variances (when valid)
    betahat = np.zeros(p)
    shat2 = np.full(p, np.inf)
    good_d = np.isfinite(d) & (d > 0)
    if np.any(good_d):
        betahat[good_d] = Xty[good_d] / d[good_d]
        if residual_variance is not None and np.isfinite(residual_variance) and residual_variance > 0:
            shat2[good_d] = residual_variance / d[good_d]

    # Normalize prior weights
    if prior_weights is None:
        prior = np.repeat(1.0 / p, p)
    else:
        prior = np.asarray(prior_weights, float)
        prior = np.maximum(prior, 0)
        sw = prior.sum()
        prior = prior / sw if sw > 0 else np.repeat(1.0 / p, p)

    # Optional optimization of V (except EM, which uses posterior moments)
    if optimize_V not in ("EM", "none"):
        V = optimize_prior_variance(optimize_V,
                                    betahat, shat2, prior,
                                    alpha=None, post_mean2=None,
                                    V_init=V,
                                    check_null_threshold=check_null_threshold, backend=backend)

    # Compute per-variable log Bayes factors and posterior responsibilities
    lbf = np.zeros(p)
    lpo = np.log(prior + np.sqrt(np.finfo(float).eps))
    mask = np.isfinite(shat2) & (shat2 > 0)
    if np.any(mask):
        denom = V + shat2[mask]
        denom = np.maximum(denom, np.finfo(float).tiny)
        lbf_mask = ((-0.5 * np.log(2 * np.pi * denom) - 0.5 * (betahat[mask]**2) / denom) -
                    (-0.5 * np.log(2 * np.pi * shat2[mask]) - 0.5 * (betahat[mask]**2) / shat2[mask]))
        lbf[mask] = lbf_mask
        lpo[mask] = lpo[mask] + lbf_mask

    # Softmax to get alpha
    m = np.max(lpo)
    w = np.exp(lpo - m)
    sw = np.sum(w)
    alpha = w / sw if sw > 0 else np.repeat(1.0 / p, p)

    # Posterior mean and variance under normal-normal model
    post_var = np.zeros(p)
    post_mean = np.zeros(p)
    if V is not None and np.isfinite(V) and V > 0 and residual_variance is not None and residual_variance > 0 and np.isfinite(residual_variance):
        denom_all = (1.0 / V) + (d / residual_variance)
        ok = np.isfinite(denom_all) & (denom_all > 0)
        post_var[ok] = 1.0 / denom_all[ok]
        post_mean[ok] = (Xty[ok] / residual_variance) * post_var[ok]
    post_mean2 = post_var + post_mean**2

    # Model evidence and log-likelihood under sigma^2
    lbf_model = float(m + np.log(sw)) if sw > 0 else -np.inf
    sigma2_safe = residual_variance if (residual_variance is not None and np.isfinite(residual_variance) and residual_variance > 0) else 1.0
    loglik = float(lbf_model + np.sum(-0.5 * np.log(2 * np.pi * sigma2_safe) - 0.5 * (y**2) / sigma2_safe))

    # EM update for V if requested
    if optimize_V == "EM":
        V = optimize_prior_variance("EM",
                                    betahat, shat2, prior,
                                    alpha=alpha, post_mean2=post_mean2,
                                    V_init=V,
                                    check_null_threshold=check_null_threshold, backend=backend)

    return dict(alpha=alpha,
                mu=post_mean,
                mu2=post_mean2,
                lbf=lbf,
                lbf_model=lbf_model,
                V=V,
                loglik=loglik)


def estimate_residual_variance_fn(X: np.ndarray, y: np.ndarray, s: Dict[str, Any], d_attr: np.ndarray) -> float:
    """
    Estimate residual variance sigma^2 as E[||y - Xb||^2] / n under the variational posterior.

    Parameters
    - X: Design matrix (n, p).
    - y: Centered response (n,).
    - s: SuSiE state dict.
    - d_attr: Column squared norms.

    Returns
    - Estimated sigma^2.
    """
    n = X.shape[0]
    ER2 = get_ER2(X, y, s, d_attr=d_attr)
    return float((1.0 / n) * ER2)


class SuSiE:
    """
    SuSiE model class (Sum of Single Effects).

    Provides a fit method to perform iterative Bayesian updates over L single-effect
    components, optionally estimating residual and prior variances, and post-processing
    results to produce credible sets and PIPs.
    """
    def __init__(self, backend: str = "numba"):
        self.backend, _ = resolve_backend(backend)
        # Posterior parameters and diagnostics (populated after fit)
        self.alpha: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.mu2: Optional[np.ndarray] = None
        self.Xr: Optional[np.ndarray] = None
        self.lbf: Optional[np.ndarray] = None
        self.lbf_variable: Optional[np.ndarray] = None

        self.intercept: float = 0.0
        self.sigma2: float = np.nan
        self.V: Optional[Union[np.ndarray, float]] = None
        self.elbo: Optional[np.ndarray] = None

        self.fitted: Optional[np.ndarray] = None
        self.sets: Optional[Dict[str, Any]] = None
        self.pip: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None

        self.niter: int = 0
        self.converged: bool = False
        self.X_column_scale_factors: Optional[np.ndarray] = None

        self.pi: Optional[np.ndarray] = None
        self.null_index: Optional[int] = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            L: Optional[int] = None,
            scaled_prior_variance: Union[float, np.ndarray] = 0.2,
            residual_variance: Optional[float] = None,
            prior_weights: Optional[np.ndarray] = None,
            null_weight: float = 0.0,
            standardize: bool = True,
            intercept: bool = True,
            estimate_residual_variance: bool = True,
            estimate_prior_variance: bool = True,
            estimate_prior_method: str = "optim",
            check_null_threshold: float = 0.0,
            prior_tol: float = 1e-9,
            residual_variance_upperbound: float = np.inf,
            s_init: Optional['SuSiE'] = None,
            coverage: Optional[float] = 0.95,
            min_abs_corr: Optional[float] = 0.5,
            compute_univariate_zscore: bool = False,
            max_iter: int = 100,
            tol: float = 1e-3,
            verbose: bool = False,
            residual_variance_lowerbound: Optional[float] = None,
            n_purity: int = 100) -> None:
        """
        Fit the SuSiE model.

        Workflow
        - Optionally augment X with a null column if null_weight != 0.
        - Initialize state and iterate over L components using SER updates.
        - Optionally update residual variance and prior variance.
        - Check ELBO convergence and construct credible sets and PIPs.

        Parameters
        - X: Design matrix (n, p0).
        - y: Response vector (n,).
        - L: Number of effects (defaults to min(10, p) if None).
        - scaled_prior_variance: Scalar or array to scale prior variance(s).
        - residual_variance: Initial sigma^2; if None, set to var(y) after centering if intercept=True.
        - prior_weights: Prior over variables (length p0). If null_weight != 0, extended accordingly.
        - null_weight: Weight of the null variable in [0, 1); adds a zero column if nonzero.
        - standardize: Center/scale X columns when computing statistics.
        - intercept: If True, fit an intercept by centering y and reconstituting fitted values.
        - estimate_residual_variance: If True, update sigma^2 each iteration.
        - estimate_prior_variance: If True, update prior variance V with method below.
        - estimate_prior_method: One of {"optim", "uniroot", "EM", "none"}.
        - check_null_threshold: Threshold favoring V=0 for SER updates.
        - prior_tol: Tolerance for pruning effects by V in PIP computation.
        - residual_variance_upperbound: Upper bound for sigma^2.
        - s_init: Optional initial state (unused in this implementation).
        - coverage: Coverage level for credible sets.
        - min_abs_corr: Minimum absolute correlation threshold for set purity.
        - compute_univariate_zscore: If True, compute per-variable univariate z-scores.
        - max_iter: Maximum number of coordinate ascent iterations.
        - tol: ELBO improvement tolerance for convergence.
        - verbose: If True, print objective values during fitting.
        - residual_variance_lowerbound: Lower bound for sigma^2; default ~ var(y)/1e4.
        - n_purity: Subsampling cap when computing set purity.

        Returns
        - None. Results are stored in the instance attributes (alpha, mu, pip, sets, etc.).
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non-finite values (NaN/Inf).")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values (NaN/Inf).")

        n, p0 = X.shape

        # Append null column if null_weight != 0
        null_used = (isinstance(null_weight, (int, float)) and (null_weight is not None) and (null_weight != 0))
        if null_used:
            if prior_weights is None:
                pw = np.repeat(1.0 / p0, p0)
            else:
                pw = np.asarray(prior_weights, float)
                pw = pw / np.sum(pw)
            if null_weight < 0 or null_weight >= 1:
                raise ValueError("null_weight must be in [0,1).")
            prior_weights = np.concatenate([pw * (1.0 - null_weight), np.array([null_weight])])
            X = np.column_stack([X, np.zeros(n)])
        p = X.shape[1]

        if L is None:
            L = min(10, p)
        else:
            L = min(L, p)

        mean_y = float(np.mean(y))
        y_cent = y.copy()
        if intercept:
            y_cent -= mean_y

        # Compute column stats for standardization and squared norms
        colstats = compute_colstats(X, center=intercept, scale=standardize)
        cm, csd, dvec = colstats["cm"], colstats["csd"], colstats["d"]

        if residual_variance_lowerbound is None:
            residual_variance_lowerbound = (np.var(y_cent, ddof=1) / 1e4) if y_cent.size > 1 else 0.0

        # Initialize and finalize state
        s = init_setup(
            n=n,
            p=p,
            L=L,
            scaled_prior_variance=scaled_prior_variance,
            residual_variance=residual_variance,
            prior_weights=prior_weights,
            null_weight=(None if not null_used else null_weight),
            varY=float(np.var(y_cent, ddof=1)) if y_cent.size > 1 else 0.0,
            standardize=standardize
        )
        s = init_finalize(s, X=X)

        self.pi = s["pi"]
        self.null_index = s.get("null_index", 0)

        # ELBO trace (index 0 reserved for -inf to simplify improvement check)
        elbo = np.full(max_iter + 1, np.nan, float)
        elbo[0] = -np.inf

        # Coordinate-wise updates over L effects
        for i in range(1, max_iter + 1):
            L_cur = s["alpha"].shape[0]
            for l in range(L_cur):
                # Remove current effect's contribution
                s["Xr"] = s["Xr"] - compute_Xb(X, s["alpha"][l, :] * s["mu"][l, :])
                R = y_cent - s["Xr"]

                # SER update for effect l
                res = single_effect_regression(
                    y=R,
                    X=X,
                    V=s["V"][l],
                    residual_variance=s["sigma2"],
                    prior_weights=s["pi"],
                    optimize_V=(estimate_prior_method if estimate_prior_variance else "none"),
                    check_null_threshold=check_null_threshold,
                    d_attr=dvec, backend=self.backend
                )

                # Update variational parameters
                s["mu"][l, :] = res["mu"]
                s["alpha"][l, :] = res["alpha"]
                s["mu2"][l, :] = res["mu2"]
                s["V"][l] = res["V"]
                s["lbf"][l] = res["lbf_model"]
                s["lbf_variable"][l, :] = res["lbf"]

                # Update KL divergence for effect l
                Eb = res["alpha"] * res["mu"]
                Eb2 = res["alpha"] * res["mu2"]
                s["KL"][l] = -res["loglik"] + SER_posterior_e_loglik(X, R, s["sigma2"], Eb, Eb2, d_attr=dvec)

                # Add current effect's updated contribution back
                s["Xr"] = s["Xr"] + compute_Xb(X, s["alpha"][l, :] * s["mu"][l, :])

            # Check ELBO improvement
            elbo[i] = get_objective(X, y_cent, s, d_attr=dvec)
            if (elbo[i] - elbo[i - 1]) < tol:
                s["converged"] = True
                break

            # Optional residual variance update
            if estimate_residual_variance:
                sig2 = estimate_residual_variance_fn(X, y_cent, s, d_attr=dvec)
                sig2 = max(residual_variance_lowerbound, sig2)
                if sig2 > residual_variance_upperbound:
                    sig2 = residual_variance_upperbound
                s["sigma2"] = sig2
                if verbose:
                    obj_now = get_objective(X, y_cent, s, d_attr=dvec)
                    print(f"objective: {obj_now:.6g}")

        # Record fit summary
        used_iter = min(i, max_iter)
        self.elbo = elbo[1:used_iter + 1]
        self.niter = used_iter
        self.converged = bool(s.get("converged", False))

        # Intercept handling and fitted values
        if intercept:
            post_mean_all = np.sum(s["alpha"] * s["mu"], axis=0)
            intercept_hat = mean_y - np.sum(cm * (post_mean_all / csd))
            s["intercept"] = float(intercept_hat)
            s["fitted"] = s["Xr"] + mean_y
        else:
            s["intercept"] = 0.0
            s["fitted"] = s["Xr"]

        # Credible sets and PIPs
        if (coverage is not None) and (min_abs_corr is not None):
            s["sets"] = susie_get_cs_py(
                res=s,
                X=X,
                Xcorr=None,
                coverage=coverage,
                min_abs_corr=min_abs_corr,
                dedup=True,
                squared=False,
                check_symmetric=True,
                n_purity=n_purity
            )
            s["pip"] = susie_get_pip_py(s, prune_by_cs=False, prior_tol=prior_tol)
            if null_used and s.get("null_index", 0) >= 0 and s["pip"] is not None:
                s["pip"] = s["pip"][:p-1]

        # Optional univariate z-scores
        if compute_univariate_zscore:
            s["z"] = calc_z(X[:, :p-1], y_cent, center=intercept, scale=standardize) if null_used \
                     else calc_z(X, y_cent, center=intercept, scale=standardize)

        s["X_column_scale_factors"] = csd

        # Export to instance attributes
        self.alpha = s["alpha"]
        self.mu = s["mu"]
        self.mu2 = s["mu2"]
        self.Xr = s["Xr"]
        self.lbf = s["lbf"]
        self.lbf_variable = s["lbf_variable"]
        self.intercept = float(s["intercept"])
        self.sigma2 = float(s["sigma2"])
        self.V = s["V"]
        self.fitted = s["fitted"]
        self.sets = s.get("sets", None)
        self.pip = s.get("pip", None)
        self.z = s.get("z", None)
        self.X_column_scale_factors = s["X_column_scale_factors"]
        self.null_index = s.get("null_index", 0)
        self.pi = s.get("pi", None)


# ==============================
# Extra: univariate regression and z-scores
# ==============================

def univariate_regression(X: np.ndarray, y: np.ndarray, Z: Optional[np.ndarray] = None,
                          center: bool = True, scale: bool = False,
                          return_residuals: bool = False) -> Dict[str, Any]:
    """
    Perform per-variable univariate regression of y on each column of X, optionally
    adjusting for covariates Z via projection.

    Parameters
    - X: Design matrix (n, p).
    - y: Response vector (n,).
    - Z: Optional covariate matrix (n, q). If provided, regress y on Z and use residuals.
    - center: If True, center y (and X/Z if applicable).
    - scale: If True, standardize X (and Z if applicable) by sample SD.
    - return_residuals: If True and Z is provided, include residuals of y after projecting out Z.

    Returns
    - Dict with keys:
      - "betahat": OLS estimates per column of X (p,).
      - "sebetahat": Standard errors per column of X (p,).
      - "residuals": Optional; residualized y when Z is provided and return_residuals=True.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).copy()
    mask = ~np.isnan(y)
    if not np.all(mask):
        X = X[mask, :]
        y = y[mask]
    if center:
        y = y - y.mean()
        if scale:
            X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0, ddof=1), 1e-15)
        else:
            X = X - X.mean(axis=0)
    else:
        if scale:
            X = X / np.maximum(X.std(axis=0, ddof=1), 1e-15)
    X = np.nan_to_num(X, nan=0.0)
    if Z is not None:
        Z = np.asarray(Z, float)
        if center:
            if scale:
                Z = (Z - Z.mean(axis=0)) / np.maximum(Z.std(axis=0, ddof=1), 1e-15)
            else:
                Z = Z - Z.mean(axis=0)
        # Project y onto the column space of Z and remove fitted component
        q, _ = np.linalg.qr(Z, mode='reduced')
        y = y - q @ (q.T @ y)

    n, p = X.shape
    betahat = np.zeros(p)
    sebetahat = np.zeros(p)
    for j in range(p):
        x = X[:, j]
        Xj = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(Xj, y, rcond=None)
        beta_j = coef[1]
        resid = y - Xj @ coef
        sig2 = np.sum(resid**2) / max(n - 2, 1)
        XtX_inv = np.linalg.inv(Xj.T @ Xj)
        se_j = np.sqrt(sig2 * XtX_inv[1, 1])
        betahat[j] = beta_j
        sebetahat[j] = se_j

    out = dict(betahat=betahat, sebetahat=sebetahat)
    if return_residuals and Z is not None:
        out["residuals"] = y
    return out

def calc_z(X: np.ndarray, Y: np.ndarray, center: bool = False, scale: bool = False) -> np.ndarray:
    """
    Compute univariate z-scores for regression of Y on each column of X.

    Parameters
    - X: Design matrix (n, p).
    - Y: Response vector (n,) or matrix (n, m). For matrix, compute z per column of Y.
    - center: Whether to center inputs as in univariate_regression.
    - scale: Whether to scale inputs as in univariate_regression.

    Returns
    - z: If Y is 1D, returns (p,) vector; else returns (p, m) matrix of z-scores.
    """
    def univariate_z(X, Y, center, scale):
        out = univariate_regression(X, Y, center=center, scale=scale)
        return out["betahat"] / out["sebetahat"]
    if Y.ndim == 1:
        return univariate_z(X, Y, center, scale)
    else:
        return np.column_stack([univariate_z(X, Y[:, i], center, scale) for i in range(Y.shape[1])])
