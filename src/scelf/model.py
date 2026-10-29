"""
scELF Negative Binomial version with unified Stage 2 Variational EM.

This version removes rho reparameterization and estimates tau directly.

Main features
-------------
1. NB observation model:
     y_m ~ NegBin(mu_m, phi), mu_m = exp(eta_m)

2. JJ quadratic lower bound gives working Gaussian quantities:
     W = diag(w_m), y* = pseudo-response

3. Working marginal Gaussian model:
     y* | b ~ N(X alpha + G_tilde b, Sigma_tau)
     Sigma_tau = W^{-1} + tau K
     K = Z Z^T

4. Collapsed variational REML objective:
     fixed effects alpha are profiled / integrated out through
     P = V - V X (X^T V X)^{-1} X^T V,
     where V = Sigma_tau^{-1}.

5. Stage 2 uses a single Variational EM loop:
   M-step:
     - update eta
     - update JJ quantities xi, W, y*
     - update alpha
     - update donor BLUP C
     - update phi by method of moments
     - update tau directly by either:
         tau.method = "bounded": scalar bounded maximization
         tau.method = "ai": score update using complete-data information n / (2 tau^2)

   E-step:
     - IBSS using collapsed REML-consistent SER updates
     - update SuSiE component prior variances

6. No damping / update step-size is used.

Refactoring notes
-----------------
This file keeps the internal algorithm and formulas unchanged.

Code organization changes:
1. Stage 1 and Stage 2 phi method-of-moments estimation now share
   the same helper function `_estimate_phi_mom`.

2. Stage 2 phi upper-bound logic is isolated in `_get_stage2_phi_max`,
   preserving the original `phi100` behavior:
     - first Stage 2 M-step:
           phi_max = min(200 * current_phi, user_phi_max)
           phi100 = True
     - later Stage 2 M-steps:
           phi_max = current_phi

3. Stage 1 warm-start logic is split into smaller helper functions:
     - `_stage1_residualize_covariates`
     - `_initialize_susie_from_residual`
     - `_update_tau_lower_bound_from_susie_prior`

Modification notes from previous version
----------------------------------------
1. After Stage 1 warm start, the adaptive lower bound of tau is set to

       tau_lower_bound = 0.01 * sum_l prior_var_l

   where prior_var_l is the SuSiE prior variance of component l.

2. If tau is not fixed, tau is projected to be at least tau_lower_bound.

3. During tau update, the effective lower bound is

       max(user_provided_tau_min, tau_lower_bound, 1e-12)

4. For tau.method = "ai", the update no longer uses average information.
   Instead, the denominator is the complete-data information

       I_comp(tau) = n / (2 tau^2).
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

from .numerics import (
    _K_dot_numba, _compute_s_vec_numba, _expand_to_cells_numba,
    _group_sum_numba, _Gt_times_vec_numba, _group_sum_matrix_numba,
    _weighted_group_sum_numba, _weighted_group_sum_matrix_numba,
)
from .results import _ResultsMixin
from .solver import ExactVSolver
from .stage1 import _Stage1Mixin
from .backend import resolve_backend, _load_cpp



# ============================================================
# Main model
# ============================================================

class ScELF(_Stage1Mixin, _ResultsMixin):
    """
    scELF model under Negative Binomial observation model.

    Parameters
    ----------
    y : array, shape (M,)
        Cell-level counts.

    Xc : array or None, shape (M, pc)
        Cell-level covariates.

    Xd : array or None, shape (n, pd)
        Donor-level covariates.

    Z : array, shape (M, n)
        Cell-to-donor incidence matrix. Each row should be one-hot.

    G : array, shape (n, p)
        Donor-level genotype matrix.

    L : int
        Number of SuSiE components.

    phi : float or None
        Initial NB size parameter. If None, Stage 1 estimates it.

    seed : int or None
        Random seed.

    tau : float or None
        If tau_fixed is None, providing tau means tau is fixed, matching the original behavior.
        If tau_fixed is explicitly False, tau is only used as initialization.

    tau_fixed : bool or None
        Whether tau is fixed throughout fitting.
        - None: tau_fixed = tau is not None
        - True: tau fixed
        - False: tau updated in Stage 2

    backend : {"auto", "numba", "cpp"}
        Use the optional native kernels when available, or explicitly select a backend.
    """

    def __init__(
        self,
        y: np.ndarray,
        Xc: Optional[np.ndarray],
        Xd: Optional[np.ndarray],
        Z: np.ndarray,
        G: np.ndarray,
        L: int,
        phi: Optional[float] = None,
        seed: Optional[int] = None,
        tau: Optional[float] = None,
        tau_fixed: Optional[bool] = None,
        backend: str = "auto",
    ) -> None:
        self.backend, _ = resolve_backend(backend)
        self.rng = np.random.default_rng(seed)

        self.y = np.asarray(y, dtype=np.float64)
        self.Xc = None if Xc is None else np.asarray(Xc, dtype=np.float64)
        self.Xd = None if Xd is None else np.asarray(Xd, dtype=np.float64)
        self.Z = np.asarray(Z)
        # Column-contiguous storage for the compiled SNP-wise reductions.
        self.G = np.asfortranarray(G, dtype=np.float64)
        self.L = int(L)

        self.M = int(self.y.shape[0])
        self.n = int(self.G.shape[0])
        self.p = int(self.G.shape[1])

        self.pc = int(self.Xc.shape[1]) if self.Xc is not None else 0
        self.pd = int(self.Xd.shape[1]) if self.Xd is not None else 0
        self.s = self.pc + self.pd

        self.cell_donor = np.argmax(self.Z, axis=1).astype(np.int64)

        self.alpha = np.zeros(self.s, dtype=np.float64) if self.s > 0 else np.zeros(0, dtype=np.float64)
        self.C = np.zeros(self.n, dtype=np.float64)

        if tau_fixed is None:
            self.tau_fixed = tau is not None
        else:
            self.tau_fixed = bool(tau_fixed)

        if tau is None:
            self.tau = 0.01
        else:
            tau_val = float(tau)
            if not np.isfinite(tau_val) or tau_val <= 0:
                raise ValueError("tau must be positive and finite.")
            self.tau = tau_val

        self.tau_lower_bound = 1e-12
        self.tau_frozen_at_lower_bound = False
        self.tau_freeze_value = None

        self.phi = float(phi) if phi is not None else np.nan

        # Keep original Stage 2 phi upper-bound behavior.
        self.phi100 = False

        self.eta = np.zeros(self.M, dtype=np.float64)
        self.mu = np.ones(self.M, dtype=np.float64)

        self.xi = np.ones(self.M, dtype=np.float64)
        self.D = np.ones(self.M, dtype=np.float64)
        self.y_work = self.y.astype(np.float64, copy=False).copy()

        self.susie_q: Dict[str, Any] = {
            "post_mean": np.zeros((self.L, self.p), dtype=np.float64),
            "post_var": np.ones((self.L, self.p), dtype=np.float64),
            "pip_comp": (
                np.full((self.L, self.p), 1.0 / max(self.p, 1), dtype=np.float64)
                if self.p > 0 and self.L > 0
                else np.zeros((self.L, 0), dtype=np.float64)
            ),
            "prior_var": np.ones(self.L, dtype=np.float64),
            "m_comp": np.zeros((self.L, self.p), dtype=np.float64),
        }

        self.m_sum = np.zeros(self.p, dtype=np.float64)

        self.pip = np.zeros(self.p, dtype=np.float64)
        self.sets: Dict[str, Any] = {}

        self.history: Dict[str, List[Any]] = {
            "F": [],
            "tau": [],
            "tau_lower_bound": [],
            "phi": [],
            "stage": [],
        }

        self._is_fitted = False

    # ============================================================
    # Public API
    # ============================================================

    def fit(
        self,
        use_covariates: bool = True,
        max_outer_iter: int = 50,
        tol_F: float = 1e-6,
        verbose: bool = True,
        stage1_options: Optional[Dict[str, Any]] = None,
        stage2_options: Optional[Dict[str, Any]] = None,
    ) -> "ScELF":
        if verbose:
            print("[Fit] Starting training.")

        self._stage1_warm_start(stage1_options or {}, verbose=verbose)

        self._stage2_variational_em(
            use_covariates=use_covariates,
            max_iter=max_outer_iter,
            tol_F=tol_F,
            options=stage2_options or {},
            verbose=verbose,
        )

        self.sets = self.get_credible_sets()
        self.pip = self.get_variant_pips(prune_by_cs=True)

        self._is_fitted = True

        if verbose:
            print("[Fit] Training completed.")

        return self


    # ============================================================
    # Unified Stage 2 Variational EM
    # ============================================================

    def _stage2_variational_em(
        self,
        use_covariates: bool,
        max_iter: int,
        tol_F: float,
        options: Dict[str, Any],
        verbose: bool = True,
    ) -> None:
        if not np.isfinite(self.phi) or self.phi <= 0:
            raise ValueError("phi must be positive before Stage 2.")

        if use_covariates and self.s > 0:
            X = self._build_design_X()
        else:
            X = np.zeros((self.M, 0), dtype=np.float64)

        max_iter = int(options.get("max_iter", max_iter))
        tol_F = float(options.get("tol_F", tol_F))

        e_step_opts = dict(options.get("e_step", {}))
        ibss_sweeps = int(e_step_opts.get("ibss_sweeps", 1))
        update_prior_var = bool(e_step_opts.get("update_prior_var", True))

        if verbose:
            tau_method = dict(options.get("tau", {})).get("method", "bounded")
            print(
                "[Stage 2: Variational EM] "
                f"max_iter={max_iter}, tol_F={tol_F}, "
                f"ibss_sweeps={ibss_sweeps}, update_prior_var={update_prior_var}, "
                f"tau_fixed={self.tau_fixed}, tau_method={tau_method}, "
                f"tau_lower_bound={self.tau_lower_bound:.6g}"
            )

        F_prev = None

        for it in range(max_iter):
            self._m_step_variational_reml(X=X, options=options)

            for _ in range(max(1, ibss_sweeps)):
                self._ibss_full_cycle_collapsed(
                    X=X,
                    update_prior_var=update_prior_var,
                )

            self.m_sum = self.susie_q["m_comp"].sum(axis=0) if self.p > 0 else np.zeros(0)

            F_new = float(self._compute_collapsed_variational_reml_elbo(X=X))

            self.history["F"].append(F_new)
            self.history["tau"].append(float(self.tau))
            self.history["tau_lower_bound"].append(float(self.tau_lower_bound))
            self.history["phi"].append(float(self.phi))
            self.history["stage"].append("stage2_variational_em")

            if verbose:
                if F_prev is None:
                    print(
                        f"[Stage 2] iter={it + 1}/{max_iter} | "
                        f"F={F_new:.6g} | phi={self.phi:.6g} | "
                        f"tau={self.tau:.6g} | tau_min={self.tau_lower_bound:.6g}"
                    )
                else:
                    dF = F_new - F_prev
                    rel_dF = dF / max(1.0, abs(F_prev))
                    print(
                        f"[Stage 2] iter={it + 1}/{max_iter} | "
                        f"F: {F_prev:.6g}->{F_new:.6g} | "
                        f"dF={dF:.3g} | rel_dF={rel_dF:.3g} | "
                        f"phi={self.phi:.6g} | tau={self.tau:.6g} | "
                        f"tau_min={self.tau_lower_bound:.6g}"
                    )

            if F_prev is not None:
                dF = F_new - F_prev
                scale = max(1.0, abs(F_prev))
                if dF <= tol_F * scale:
                    if verbose:
                        print("[Stage 2] Stop: ELBO increase is below tolerance.")
                    break

            F_prev = F_new

    def _m_step_variational_reml(self, X: np.ndarray, options: Dict[str, Any]) -> None:
        """
        One M-step under current q.

        Updates:
          eta, xi, W, y*, alpha, C, phi, tau.
        """
        phi_opts = dict(options.get("phi", {}))
        update_phi = bool(phi_opts.get("update", True))
        phi_min = float(phi_opts.get("phi_min", 1e-4))
        phi_init = float(phi_opts.get("phi_init", 1e4))

        phi_max = self._get_stage2_phi_max(phi_opts)

        self._update_linear_predictor(include_genetics=True)
        self._update_jj_working_quantities()

        solver = self._make_exact_solver()
        r_g = self._current_r_g()
        self._update_fixed_effects(solver=solver, X=X, r_g=r_g)
        self._update_random_effects(W=self.D, X=X, r_g=r_g)

        self._update_linear_predictor(include_genetics=True)

        if update_phi:
            self._update_phi_mom(
                phi_min=phi_min,
                phi_max=phi_max,
                phi_init=phi_init,
            )

        self._update_jj_working_quantities()

        if not self.tau_fixed:
            if self.tau_frozen_at_lower_bound:
                self.tau = float(self.tau_freeze_value)
            else:
                self._update_tau_direct(X=X, options=options)

        solver = self._make_exact_solver()
        r_g = self._current_r_g()
        self._update_fixed_effects(solver=solver, X=X, r_g=r_g)
        self._update_random_effects(W=self.D, X=X, r_g=r_g)

        self._update_linear_predictor(include_genetics=True)
        self._update_jj_working_quantities()

    # ============================================================
    # Design matrix and linear predictor
    # ============================================================

    def _build_design_X(self) -> np.ndarray:
        parts = []

        if self.Xc is not None and self.pc > 0:
            if self.Xc.shape[0] != self.M:
                raise ValueError(f"Xc has inconsistent rows: {self.Xc.shape[0]} vs M={self.M}.")
            parts.append(self.Xc.astype(np.float64, copy=False))

        if self.Xd is not None and self.pd > 0:
            if self.Xd.shape[0] != self.n:
                raise ValueError(f"Xd has inconsistent rows: {self.Xd.shape[0]} vs n={self.n}.")
            ZX_cols = [
                _expand_to_cells_numba(self.Xd[:, j].astype(np.float64, copy=False), self.cell_donor, self.M)
                for j in range(self.pd)
            ]
            parts.append(np.stack(ZX_cols, axis=1).astype(np.float64, copy=False))

        if len(parts) == 0:
            return np.zeros((self.M, 0), dtype=np.float64)

        X = np.concatenate(parts, axis=1).astype(np.float64, copy=False)

        if X.shape[1] != self.s:
            self.s = X.shape[1]
            self.alpha = np.zeros(self.s, dtype=np.float64)

        return X

    def _update_linear_predictor(self, include_genetics: bool = True) -> None:
        X = self._build_design_X()
        eta = np.zeros(self.M, dtype=np.float64)

        if X.shape[1] > 0:
            eta += X @ self.alpha

        if include_genetics and self.p > 0:
            donor_g_eff = self.G @ self.m_sum
            eta += _expand_to_cells_numba(donor_g_eff.astype(np.float64, copy=False), self.cell_donor, self.M)

        eta += _expand_to_cells_numba(self.C.astype(np.float64, copy=False), self.cell_donor, self.M)

        self.eta = eta
        self.mu = np.exp(np.clip(self.eta, -100.0, 100.0))

    # ============================================================
    # JJ bound and phi update
    # ============================================================

    @staticmethod
    def _lambda_jj(xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        out = np.empty_like(xi)

        small = xi < 1e-8
        out[small] = 0.125

        x = xi[~small]
        out[~small] = np.tanh(0.5 * x) / (4.0 * x)

        return np.maximum(out, 1e-18)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.where(x > 50.0, x, np.log1p(np.exp(x)))

    def _update_jj_working_quantities(self) -> None:
        if not np.isfinite(self.phi) or self.phi <= 0:
            raise ValueError("phi must be positive for JJ update.")

        y = self.y.astype(np.float64, copy=False)
        phi = float(self.phi)
        log_phi = float(np.log(phi))

        xi = np.abs(self.eta - log_phi)
        lam = self._lambda_jj(xi)

        w = 2.0 * (y + phi) * lam
        w = np.maximum(w, 1e-8)

        y_star = log_phi + (y - phi) / (2.0 * w)

        self.xi = xi.astype(np.float64, copy=False)
        self.D = w.astype(np.float64, copy=False)
        self.y_work = y_star.astype(np.float64, copy=False)

    def _update_phi_mom(
        self,
        phi_min: float = 1e-4,
        phi_max: float = 1e4,
        phi_init: float = 1e4,
    ) -> None:
        self.phi = self._estimate_phi_mom(
            phi_min=phi_min,
            phi_max=phi_max,
            phi_init=phi_init,
        )

    # ============================================================
    # Exact marginal precision solver
    # ============================================================


    def _make_exact_solver(self) -> "ExactVSolver":
        return ExactVSolver(
            W=self.D.astype(np.float64, copy=False),
            tau=float(self.tau),
            cell_donor=self.cell_donor,
            n=self.n,
        )

    # ============================================================
    # Projection matrix P
    # ============================================================

    def _make_projection_cache(
        self,
        solver: "ExactVSolver",
        X: np.ndarray,
    ) -> Dict[str, Any]:
        s = X.shape[1] if X is not None else 0

        if s == 0:
            return {
                "s": 0,
                "VX": None,
                "XtVX": np.zeros((0, 0), dtype=np.float64),
                "Ainv": np.zeros((0, 0), dtype=np.float64),
                "logdet_XtVX": 0.0,
            }

        VX = solver.solve_matrix(X)

        XtVX = X.T @ VX
        Ainv, XtVX_reg, logdet = self._factor_projection_gram(XtVX)
        return {
            "s": s,
            "VX": VX,
            "XtVX": XtVX_reg,
            "Ainv": Ainv,
            "logdet_XtVX": logdet,
            "T": _group_sum_matrix_numba(VX, self.cell_donor, self.n),
        }

    @staticmethod
    def _factor_projection_gram(XtVX: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Keep the original regularization and singular-matrix fallback."""
        s = XtVX.shape[0]
        reg = XtVX + 1e-10 * np.eye(s, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(reg)
        if sign <= 0 or not np.isfinite(logdet):
            eig = np.linalg.eigvalsh(0.5 * (reg + reg.T))
            logdet = float(np.sum(np.log(np.maximum(eig, 1e-300))))
        try:
            Ainv = np.linalg.inv(reg)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(reg)
        return Ainv, reg, float(logdet)

    def _apply_donor_P(self, v: np.ndarray, solver: ExactVSolver, cache: Dict[str, Any]) -> np.ndarray:
        """H v with H = Z.T P Z; never construct H or a cell vector."""
        u = solver.w_donor * solver.c_coef
        out = u * v if v.ndim == 1 else u[:, None] * v
        if cache["s"]:
            T = cache["T"]
            out -= T @ (cache["Ainv"] @ (T.T @ v))
        return out

    def _cpp_projection_arrays(self, cache: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Cache tiny layout conversions; genotype data are already column-contiguous."""
        if "cpp_T" not in cache:
            cache["cpp_T"] = np.asfortranarray(cache["T"]) if cache["s"] else np.empty((self.n, 0), order="F")
            cache["cpp_Ainv"] = np.asfortranarray(cache["Ainv"]) if cache["s"] else np.empty((0, 0), order="F")
        return cache["cpp_T"], cache["cpp_Ainv"]

    def _apply_P(
        self,
        v: np.ndarray,
        solver: "ExactVSolver",
        X: np.ndarray,
        cache: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if cache is None:
            cache = self._make_projection_cache(solver, X)

        q = solver.solve(v.astype(np.float64, copy=False))

        if cache["s"] == 0:
            return q

        VX = cache["VX"]
        Ainv = cache["Ainv"]

        c = X.T @ q
        return q - VX @ (Ainv @ c)

    # ============================================================
    # Fixed effects and random effects
    # ============================================================

    def _current_r_g(self) -> np.ndarray:
        """
        r_g = y* - G_tilde m.
        """
        if self.p > 0:
            donor_g = self.G @ self.m_sum
            return self.y_work - _expand_to_cells_numba(donor_g.astype(np.float64, copy=False), self.cell_donor, self.M)
        return self.y_work.copy()

    def _update_fixed_effects(
        self,
        solver: "ExactVSolver",
        X: np.ndarray,
        r_g: np.ndarray,
    ) -> None:
        if X is None or X.shape[1] == 0:
            return

        s = X.shape[1]

        VX = solver.solve_matrix(X)

        XtVX = X.T @ VX
        Vr = solver.solve(r_g)
        rhs = X.T @ Vr

        XtVX_reg = XtVX + 1e-10 * np.eye(s, dtype=np.float64)

        try:
            alpha_new = np.linalg.solve(XtVX_reg, rhs)
        except np.linalg.LinAlgError:
            alpha_new = np.linalg.lstsq(XtVX_reg, rhs, rcond=None)[0]

        self.alpha = alpha_new.astype(np.float64, copy=False)

    def _update_random_effects(
        self,
        W: np.ndarray,
        X: np.ndarray,
        r_g: np.ndarray,
    ) -> None:
        W = np.maximum(W.astype(np.float64, copy=False), 1e-12)

        if X is not None and X.shape[1] > 0:
            u = r_g - X @ self.alpha
        else:
            u = r_g.copy()

        Wu = W * u
        s_w = _group_sum_numba(W, self.cell_donor, self.n)
        rhs = _group_sum_numba(Wu.astype(np.float64, copy=False), self.cell_donor, self.n)

        denom = s_w + 1.0 / max(float(self.tau), 1e-12)
        denom = np.maximum(denom, 1e-12)

        self.C = (rhs / denom).astype(np.float64, copy=False)

    # ============================================================
    # Direct tau update
    # ============================================================

    def _update_tau_direct(self, X: np.ndarray, options: Dict[str, Any]) -> None:
        """
        Directly update tau.

        options["tau"]["method"]:
          - "bounded": one-dimensional bounded scalar maximization
          - "ai": score update using complete-data information n / (2 tau^2)

        The effective lower bound is

            max(options["tau"]["tau_min"], self.tau_lower_bound, 1e-12).

        Once tau reaches the lower bound during update, tau is frozen at that
        lower bound for all subsequent iterations.
        """
        tau_opts = dict(options.get("tau", {}))

        method = str(tau_opts.get("method", "bounded")).lower()

        user_tau_min = float(tau_opts.get("tau_min", 1e-8))
        adaptive_tau_min = float(getattr(self, "tau_lower_bound", 1e-12))
        tau_min = max(user_tau_min, adaptive_tau_min, 1e-12)

        tau_max = float(tau_opts.get("tau_max", 1e8))
        tau_max = max(tau_max, tau_min * 10.0)

        if self.tau_frozen_at_lower_bound:
            if self.tau_freeze_value is None:
                self.tau_freeze_value = float(tau_min)
            self.tau = float(self.tau_freeze_value)
            return

        def freeze_tau_at_lower_bound() -> None:
            self.tau = float(tau_min)
            self.tau_frozen_at_lower_bound = True
            self.tau_freeze_value = float(tau_min)

        if method in ("bounded", "scalar", "brent"):
            xatol = float(tau_opts.get("xatol", 1e-5))

            boundary_tol = float(
                tau_opts.get(
                    "boundary_tol",
                    max(1e-10, xatol * max(1.0, tau_min)),
                )
            )

            statistics = self._prepare_tau_statistics(X)

            def obj(tau: float) -> float:
                q = self._collapsed_variational_reml_Q_tau(X=X, tau=tau, statistics=statistics)
                return -q

            res = minimize_scalar(
                obj,
                bounds=(tau_min, tau_max),
                method="bounded",
                options={"xatol": xatol},
            )

            if res.success and np.isfinite(res.x):
                tau_hat = float(np.clip(res.x, tau_min, tau_max))
            else:
                tau_hat = float(np.clip(self.tau, tau_min, tau_max))

            if tau_hat <= tau_min + boundary_tol:
                freeze_tau_at_lower_bound()
            else:
                self.tau = tau_hat

            return

        if method in ("ai", "aireml", "ai_reml"):
            max_ai_iter = int(tau_opts.get("ai_max_iter", 5))
            trace_samples = int(tau_opts.get("trace_samples", 16))

            tau = float(np.clip(self.tau, tau_min, tau_max))

            for _ in range(max_ai_iter):
                score_tau, info_tau = self._score_complete_info_tau_variational(
                    X=X,
                    tau=tau,
                    trace_samples=trace_samples,
                )

                if not np.isfinite(score_tau) or not np.isfinite(info_tau) or info_tau <= 1e-16:
                    break

                tau_raw = tau + score_tau / info_tau

                if tau_raw <= tau_min:
                    freeze_tau_at_lower_bound()
                    return

                tau = float(np.clip(tau_raw, tau_min, tau_max))

                if tau <= tau_min:
                    freeze_tau_at_lower_bound()
                    return

            self.tau = tau
            return

        raise ValueError(f"Unknown tau update method: {method}")

    def _prepare_trace_statistics(self) -> Dict[str, np.ndarray]:
        """Fixed-q quantities shared by trace evaluations at different tau."""
        if self.p == 0 or self.L == 0:
            return {"d": np.zeros(self.p), "U": np.zeros((self.n, 0))}
        q = self.susie_q
        d = np.sum(q["pip_comp"] * (q["post_mean"] ** 2 + q["post_var"]), axis=0)
        U = self.G @ q["m_comp"].T
        return {"d": d, "U": U}

    def _prepare_tau_statistics(self, X: np.ndarray) -> Dict[str, Any]:
        """Weighted within/between-donor statistics for one fixed-W, fixed-q search.

        Centering within donors avoids subtracting nearly equal weighted Gram
        matrices when tau is large. All cell-sized work happens once per search.
        """
        W = np.maximum(self.D, 1e-12)
        w = _group_sum_numba(W, self.cell_donor, self.n)
        inv_w = np.zeros_like(w)
        np.divide(1.0, w, out=inv_w, where=w > 0)
        z = self.y_work.copy()
        if self.p:
            z -= _expand_to_cells_numba(self.G @ self.m_sum, self.cell_donor, self.M)
        wz = _weighted_group_sum_numba(z, W, self.cell_donor, self.n)
        wx = _weighted_group_sum_matrix_numba(X, W, self.cell_donor, self.n)
        zc = z - (wz * inv_w)[self.cell_donor]
        Xc = X - (wx * inv_w[:, None])[self.cell_donor]
        Wzc = W * zc
        return {
            "w": w, "inv_w": inv_w, "wz": wz, "wx": wx,
            "within_xx": Xc.T @ (W[:, None] * Xc),
            "within_xz": Xc.T @ Wzc,
            "within_zz": float(zc @ Wzc),
            "logdet_Winv": -float(np.log(W).sum()),
            "trace": self._prepare_trace_statistics(),
        }

    def _collapsed_variational_reml_Q_tau(
        self, X: np.ndarray, tau: float, statistics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Exact tau objective from sufficient statistics; does not mutate self.tau."""
        if statistics is None:
            statistics = self._prepare_tau_statistics(X)
        st = statistics
        tau = float(max(tau, 1e-12))
        c = 1.0 / np.maximum(1.0 + tau * st["w"], 1e-12)
        between = c * st["inv_w"]
        wx, wz = st["wx"], st["wz"]
        quad = st["within_zz"] + float(wz @ (between * wz))
        count = X.shape[1]
        logdet_XtVX = 0.0
        cache = {"s": count}
        if count:
            gram = st["within_xx"] + wx.T @ (between[:, None] * wx)
            Ainv, _, logdet_XtVX = self._factor_projection_gram(gram)
            rhs = st["within_xz"] + wx.T @ (between * wz)
            quad -= float(rhs @ (Ainv @ rhs))
            cache.update(Ainv=Ainv, T=c[:, None] * wx)
        u = st["w"] * c
        if self.backend == "cpp":
            T, Ainv = self._cpp_projection_arrays(cache)
            trPB = _load_cpp().trace_gpg(
                self.G, u, T, Ainv, st["trace"]["d"], np.asfortranarray(st["trace"]["U"])
            )
        else:
            sP = self._diag_donor_GPG(u, cache)
            trPB = self._trace_from_donor_statistics(u, cache, sP, st["trace"])
        logdet_sigma = st["logdet_Winv"] + float(np.log1p(tau * st["w"]).sum())
        return -0.5 * (logdet_sigma + logdet_XtVX + quad + trPB)

    def _score_complete_info_tau_variational(
        self,
        X: np.ndarray,
        tau: float,
        trace_samples: int = 16,
    ) -> Tuple[float, float]:
        """
        Score update for tau under the collapsed variational objective,
        using complete-data information instead of average information.

        Score:
          0.5 * [
              zbar^T P K P zbar
              + tr(P K P B_q)
              - tr(P K)
          ]

        Complete-data information:
          n / (2 tau^2)
        """
        old_tau = float(self.tau)
        self.tau = float(max(tau, 1e-12))

        try:
            solver = self._make_exact_solver()
            cache = self._make_projection_cache(solver, X)

            zbar = self.y_work.copy()
            if self.p > 0:
                gm = self.G @ self.m_sum
                zbar -= _expand_to_cells_numba(gm.astype(np.float64, copy=False), self.cell_donor, self.M)

            u = self._apply_P(zbar, solver, X, cache)
            Ku = solver.dotK(u)

            quad_score = float(np.dot(u, Ku))

            tr_PK = self._trace_PK_exact(solver=solver, X=X, cache=cache)

            tr_PKPB = self._hutchinson_trace(
                A_mv=lambda z: self._apply_P(
                    solver.dotK(
                        self._apply_P(
                            self._apply_Bq(z),
                            solver,
                            X,
                            cache,
                        )
                    ),
                    solver,
                    X,
                    cache,
                ),
                size=self.M,
                n_samples=trace_samples,
            )

            score_tau = 0.5 * (quad_score + tr_PKPB - tr_PK)

            tau_safe = max(float(tau), 1e-12)
            info_tau = float(self.n) / (2.0 * tau_safe ** 2)
            info_tau = max(float(info_tau), 1e-12)

            return float(score_tau), float(info_tau)
        finally:
            self.tau = old_tau

    # ============================================================
    # Collapsed REML-consistent SuSiE E-step
    # ============================================================

    def _ibss_full_cycle_collapsed(
        self,
        X: np.ndarray,
        update_prior_var: bool = True,
    ) -> None:
        """
        One full IBSS cycle over all SuSiE components.

        Uses SER updates based on:
          a_j = g_j^T P r_l
          s_j = g_j^T P g_j

        where P is the collapsed REML projection matrix.
        """
        if self.p == 0 or self.L == 0:
            return

        solver = self._make_exact_solver()
        cache = self._make_projection_cache(solver, X)

        sP_vec = self._compute_diag_GPG(solver=solver, X=X, cache=cache)
        sP_vec = np.maximum(sP_vec, 1e-12)

        # b = Z.T P y*: one cell-level projection per sweep, not per effect.
        b = _group_sum_numba(
            self._apply_P(self.y_work, solver, X, cache), self.cell_donor, self.n
        )
        if self.backend == "cpp":
            T, Ainv = self._cpp_projection_arrays(cache)
            q = self.susie_q
            _load_cpp().ibss_sweep(
                self.G, solver.w_donor * solver.c_coef, T, Ainv, b, sP_vec,
                q["pip_comp"], q["post_mean"], q["post_var"], q["m_comp"],
                q["prior_var"], self.m_sum, update_prior_var,
            )
            return
        for l in range(self.L):
            m_l_old = self.susie_q["m_comp"][l, :].copy()
            m_minus_l = self.m_sum - m_l_old
            h = self.G @ m_minus_l
            zt_Pr = b - self._apply_donor_P(h, solver, cache)
            a = _Gt_times_vec_numba(self.G, zt_Pr)

            sigma2 = float(self.susie_q["prior_var"][l])
            sigma2 = max(sigma2, 1e-12)

            inv_sigma2 = 1.0 / sigma2

            s2 = 1.0 / np.maximum(sP_vec + inv_sigma2, 1e-20)
            s2 = np.maximum(s2, 1e-18)

            mu = s2 * a

            log_prior = -math.log(float(max(self.p, 1)))
            logits = (
                log_prior
                + 0.5 * (np.log(s2) - np.log(sigma2))
                + 0.5 * (mu ** 2 / np.maximum(s2, 1e-18))
            )

            logits -= np.max(logits)
            weights = np.exp(logits)
            norm = float(np.sum(weights))

            if not np.isfinite(norm) or norm <= 0:
                pip = np.full(self.p, 1.0 / self.p, dtype=np.float64)
            else:
                pip = weights / norm

            pip = np.clip(pip, 0.0, 1.0)
            pip = pip / max(float(pip.sum()), 1e-20)

            self.susie_q["pip_comp"][l, :] = pip
            self.susie_q["post_mean"][l, :] = mu
            self.susie_q["post_var"][l, :] = s2

            m_l_new = pip * mu
            self.susie_q["m_comp"][l, :] = m_l_new
            self.m_sum = m_minus_l + m_l_new

            if update_prior_var:
                sigma2_new = float(np.sum(pip * (mu ** 2 + s2)))
                self.susie_q["prior_var"][l] = max(sigma2_new, 1e-12)

    def _compute_diag_GPG(
        self,
        solver: "ExactVSolver",
        X: np.ndarray,
        cache: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute diag(G_tilde^T P G_tilde).
        """
        if "diag_GPG" not in cache:
            cache["diag_GPG"] = self._diag_donor_GPG(solver.w_donor * solver.c_coef, cache)
        return cache["diag_GPG"]

    def _diag_donor_GPG(self, u: np.ndarray, cache: Dict[str, Any]) -> np.ndarray:
        if self.backend == "cpp":
            T, Ainv = self._cpp_projection_arrays(cache)
            return _load_cpp().diag_gpg(self.G, u, T, Ainv)
        sV = _compute_s_vec_numba(self.G, u)
        if cache["s"]:
            B = (self.G.T @ cache["T"]).T
            sV -= np.sum(B * (cache["Ainv"] @ B), axis=0)
        return np.maximum(sV, 1e-12)

    @staticmethod
    def _trace_from_donor_statistics(
        u: np.ndarray, cache: Dict[str, Any], sP: np.ndarray, statistics: Dict[str, np.ndarray],
    ) -> float:
        U = statistics["U"]
        HU = u[:, None] * U
        if cache["s"]:
            T = cache["T"]
            HU -= T @ (cache["Ainv"] @ (T.T @ U))
        out = float(statistics["d"] @ sP) - float(np.sum(U * HU))
        return float(max(out, 0.0))

    # ============================================================
    # Collapsed variational REML ELBO
    # ============================================================

    def _compute_collapsed_variational_reml_elbo(self, X: np.ndarray) -> float:
        solver = self._make_exact_solver()
        cache = self._make_projection_cache(solver, X)

        C_JJ = self._compute_C_JJ()

        zbar = self.y_work.copy()
        if self.p > 0:
            gm = self.G @ self.m_sum
            zbar -= _expand_to_cells_numba(gm.astype(np.float64, copy=False), self.cell_donor, self.M)

        Pz = self._apply_P(zbar, solver, X, cache)
        quad = float(np.dot(zbar, Pz))

        logdet_sigma = solver.logdet_sigma()
        logdet_XtVX = cache["logdet_XtVX"]

        trPB = self._trace_P_Bq_exact(solver=solver, X=X, cache=cache)
        KL = self._kl_sum()

        df = self.M - int(cache["s"])
        const = df * math.log(2.0 * math.pi)

        F = C_JJ - 0.5 * (logdet_sigma + logdet_XtVX + quad + trPB + const) - KL
        return float(F)

    def _compute_C_JJ(self) -> float:
        y = self.y.astype(np.float64, copy=False)
        phi = float(self.phi)
        xi = np.maximum(self.xi.astype(np.float64, copy=False), 0.0)

        lam = self._lambda_jj(xi)
        w = np.maximum(self.D.astype(np.float64, copy=False), 1e-12)

        sp = self._softplus(xi)

        term_gamma = gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0)
        term_bound = -(y + phi) * (sp - 0.5 * xi - lam * xi ** 2)
        denom = 16.0 * (y + phi) * np.maximum(lam, 1e-18)
        term_quad = (y - phi) ** 2 / np.maximum(denom, 1e-18)
        term_logw = -0.5 * np.log(w)

        cjj = term_gamma + term_bound + term_quad + term_logw
        cjj = np.nan_to_num(cjj, nan=0.0, posinf=0.0, neginf=0.0)

        return float(np.sum(cjj))

    def _trace_P_Bq_exact(
        self,
        solver: "ExactVSolver",
        X: np.ndarray,
        cache: Dict[str, Any],
    ) -> float:
        """
        trace(P B_q), where

          B_q = G_tilde Sigma_b G_tilde^T

        and

          Sigma_b = sum_l Cov_q(b_l).
        """
        if self.p == 0 or self.L == 0:
            return 0.0

        if self.backend == "cpp":
            T, Ainv = self._cpp_projection_arrays(cache)
            statistics = self._prepare_trace_statistics()
            return _load_cpp().trace_gpg(
                self.G, solver.w_donor * solver.c_coef, T, Ainv,
                statistics["d"], np.asfortranarray(statistics["U"]),
            )
        sP_vec = self._compute_diag_GPG(solver, X, cache)
        return self._trace_from_donor_statistics(
            solver.w_donor * solver.c_coef, cache, sP_vec, self._prepare_trace_statistics()
        )

    def _kl_sum(self) -> float:
        """
        KL sum of SuSiE variational posterior against SNP-uniform single-effect prior.
        """
        if self.p == 0 or self.L == 0:
            return 0.0

        KL_sum = 0.0
        log_pi = -math.log(float(self.p))

        for l in range(self.L):
            alpha_l = np.asarray(self.susie_q["pip_comp"][l, :], dtype=np.float64)
            mu_l = np.asarray(self.susie_q["post_mean"][l, :], dtype=np.float64)
            s2_l = np.asarray(self.susie_q["post_var"][l, :], dtype=np.float64)
            sigma2_l = float(self.susie_q["prior_var"][l])
            sigma2_l = max(sigma2_l, 1e-12)

            alpha_clip = np.maximum(alpha_l, 1e-300)
            s2_clip = np.maximum(s2_l, 1e-18)

            kl_cat = float(np.sum(alpha_l * (np.log(alpha_clip) - log_pi)))

            kl_gauss = 0.5 * float(
                np.sum(
                    alpha_l
                    * (
                        (mu_l ** 2 + s2_clip) / sigma2_l
                        - 1.0
                        - np.log(s2_clip / sigma2_l)
                    )
                )
            )

            KL_sum += kl_cat + kl_gauss

        return float(KL_sum)

    # ============================================================
    # Auxiliary operations for tau score update
    # ============================================================

    def _apply_Bq(self, v: np.ndarray) -> np.ndarray:
        """
        Apply B_q = G_tilde Sigma_b G_tilde^T to a cell-level vector v.
        """
        if self.p == 0 or self.L == 0:
            return np.zeros_like(v, dtype=np.float64)

        donor_v = _group_sum_numba(v.astype(np.float64, copy=False), self.cell_donor, self.n)
        t = self.G.T @ donor_v

        out_p = np.zeros(self.p, dtype=np.float64)

        for l in range(self.L):
            alpha_l = self.susie_q["pip_comp"][l, :]
            mu_l = self.susie_q["post_mean"][l, :]
            s2_l = self.susie_q["post_var"][l, :]
            m_l = self.susie_q["m_comp"][l, :]

            d_l = alpha_l * (mu_l ** 2 + s2_l)
            out_p += d_l * t - m_l * float(np.dot(m_l, t))

        donor_out = self.G @ out_p
        return _expand_to_cells_numba(donor_out.astype(np.float64, copy=False), self.cell_donor, self.M)

    def _trace_PK_exact(
        self,
        solver: "ExactVSolver",
        X: np.ndarray,
        cache: Dict[str, Any],
    ) -> float:
        """
        trace(P K), K = Z Z^T.
        """
        tr_VK = float(np.sum(solver.w_donor * solver.c_coef))

        if cache["s"] == 0:
            return tr_VK

        T = cache["T"]
        correction = float(np.trace(cache["Ainv"] @ (T.T @ T)))
        return float(tr_VK - correction)

    def _hutchinson_trace(
        self,
        A_mv,
        size: int,
        n_samples: int = 16,
    ) -> float:
        if size <= 0:
            return 0.0

        n_samples = max(int(n_samples), 1)
        acc = 0.0

        for _ in range(n_samples):
            z = self.rng.integers(0, 2, size=size, dtype=np.int8)
            z = 2.0 * z.astype(np.float64, copy=False) - 1.0
            Az = A_mv(z)
            acc += float(np.dot(z, Az))

        return acc / float(n_samples)


