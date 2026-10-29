"""PIP-based colocalization conditional on H3 (distinct) versus H4 (shared).

Align variants before calling :func:`coloc_pip`. This function does not align
alleles, select credible sets, or distribute unreported posterior mass.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


def _logsumexp(a: np.ndarray) -> float:
    """Stable log(sum(exp(a)))."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = np.max(a)
    if np.isneginf(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def _logsubexp(x: float, y: float) -> float:
    """
    Stable log(exp(x) - exp(y)).
    Returns -inf if exp(x) <= exp(y) (i.e., y >= x).
    """
    if not np.isfinite(x):
        return -np.inf
    if not np.isfinite(y):
        # if y is -inf, exp(y)=0 => log(exp(x)-0)=x
        return x
    if y >= x:
        return -np.inf
    # x + log(1 - exp(y-x))
    return float(x + np.log1p(-np.exp(y - x)))


@dataclass(frozen=True)
class ColocPIPResult:
    """Result for coloc-pip restricted to H3 vs H4 (as in gentropy ColocPIP)."""
    h3: float
    h4: float
    score: float  # normalized score (here == h4)
    log_pp3: float  # unnormalized log weight for H3
    log_pp4: float  # unnormalized log weight for H4


def coloc_pip(
    pip1: np.ndarray,
    pip2: np.ndarray,
    p1: float = 1e-4,
    p2: float = 1e-4,
    p12: float = 1e-7,
    pseudocount: float = 1e-16,
    validate: bool = True,
) -> ColocPIPResult:
    """
    Calculate the PIP-product H3/H4 score used by the analysis workflows.

    - Input: two aligned PIP vectors (same length, same variant order).
    - Computes only H3 vs H4, then normalizes so H3 + H4 = 1.
    - Returns (H3, H4, score=H4) plus the unnormalized log-weights.
    - The score is conditional on H3 or H4, not a five-hypothesis coloc posterior.
    - Defaults retain the manuscript analysis prior weights; pass other priors
      explicitly when comparing to another implementation or release.

    Notes:
    - Works best when each PIP vector corresponds to a *single signal* within the locus.
    - If your PIPs sum to 1 across the region, then:
        sum_log_pip1 = log(sum(pip1)) = 0
        sum_log_pip2 = log(sum(pip2)) = 0
        log_sum_both  = log(sum(pip1 * pip2))
        logdiff        = log(1 - sum(pip1 * pip2))  (up to numerical tolerance)
    """
    pip1 = np.asarray(pip1, dtype=np.float64)
    pip2 = np.asarray(pip2, dtype=np.float64)

    if validate:
        if pip1.shape != pip2.shape:
            raise ValueError(f"pip1 and pip2 must have same shape, got {pip1.shape} vs {pip2.shape}")
        if pip1.ndim != 1:
            raise ValueError(f"pip1/pip2 must be 1D arrays, got ndim={pip1.ndim}")
        if pip1.size == 0:
            raise ValueError("PIP vectors must be non-empty.")
        if np.any(~np.isfinite(pip1)) or np.any(~np.isfinite(pip2)):
            raise ValueError("pip1/pip2 contain non-finite values (nan/inf).")
        if np.any((pip1 < 0) | (pip1 > 1)) or np.any((pip2 < 0) | (pip2 > 1)):
            raise ValueError("PIPs must lie in [0, 1].")
        if pip1.sum() <= 0 or pip2.sum() <= 0:
            raise ValueError("Each PIP vector must contain positive posterior support.")
        if not all(np.isfinite(p) and 0 < p <= 1 for p in (p1, p2, p12)):
            raise ValueError("Prior weights must be finite and in (0, 1].")
        if not np.isfinite(pseudocount) or not 0 < pseudocount < 1:
            raise ValueError("pseudocount must be finite and in (0, 1).")

    # Match gentropy: priors are lower-bounded to avoid log(0)
    p1 = max(float(p1), pseudocount)
    p2 = max(float(p2), pseudocount)
    p12 = max(float(p12), pseudocount)

    # Match gentropy: ensure no PIP is exactly zero
    pip1 = np.maximum(pip1, pseudocount)
    pip2 = np.maximum(pip2, pseudocount)

    # Log-space
    log_pip1 = np.log(pip1)
    log_pip2 = np.log(pip2)

    # log(sum(pip)) via logsumexp(log(pip))
    sum_log_pip1 = _logsumexp(log_pip1)                 # = log(sum(pip1))
    sum_log_pip2 = _logsumexp(log_pip2)                 # = log(sum(pip2))
    log_sum_both = _logsumexp(log_pip1 + log_pip2)      # = log(sum(pip1 * pip2))

    # logdiff = log( exp(x) - exp(y) ) where:
    # x = log(sum(pip1)) + log(sum(pip2))
    # y = log(sum(pip1*pip2))
    x = sum_log_pip1 + sum_log_pip2
    y = log_sum_both
    logdiff = _logsubexp(x, y)  # -inf if y>=x

    # Unnormalized log-weights (gentropy names them PP3/PP4 but they are in log-space)
    log_pp3 = np.log(p1) + np.log(p2) + logdiff
    log_pp4 = np.log(p12) + log_sum_both

    # Normalize between H3 and H4 only (as in gentropy ColocPIP)
    denom = _logsumexp(np.array([log_pp3, log_pp4], dtype=np.float64))
    if np.isneginf(denom):
        # Degenerate case: both weights ~ 0
        h3 = 0.5
        h4 = 0.5
    else:
        h3 = float(np.exp(log_pp3 - denom)) if np.isfinite(log_pp3) else 0.0
        h4 = float(np.exp(log_pp4 - denom)) if np.isfinite(log_pp4) else 0.0

    # In this 2-hypothesis normalization, score is simply H4
    return ColocPIPResult(h3=h3, h4=h4, score=h4, log_pp3=float(log_pp3), log_pp4=float(log_pp4))


