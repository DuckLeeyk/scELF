"""Stage 1 warm-start methods for scelf.ScELF."""

from typing import Any, Dict, List, Tuple

import numpy as np
import statsmodels.api as sm

from .numerics import _group_sum_numba
from .susie import SuSiE

class _Stage1Mixin:
    # ============================================================
    # Shared phi helpers
    # ============================================================

    def _estimate_phi_mom(
        self,
        phi_min: float = 1e-4,
        phi_max: float = 1e4,
        phi_init: float = 1e4,
    ) -> float:
        """
        Method-of-moments estimator for NB size parameter phi.

        This helper preserves the original formula and clipping behavior:

            phi_hat = sum(mu^2) / sum((y - mu)^2 - mu)

        with fallback to phi_init or phi_max under the same conditions as before.
        """
        y = self.y.astype(np.float64)
        mu = np.maximum(np.exp(np.clip(self.eta, -100.0, 100.0)), 1e-12)

        num = float(np.sum(mu ** 2))
        denom = float(np.sum((y - mu) ** 2 - mu))

        if not np.isfinite(num) or not np.isfinite(denom):
            phi_hat = phi_init
        elif denom <= 0:
            phi_hat = phi_max
        else:
            phi_hat = num / denom

        if not np.isfinite(phi_hat) or phi_hat <= 0:
            phi_hat = phi_init

        return float(np.clip(phi_hat, phi_min, phi_max))

    def _get_stage2_phi_max(self, phi_opts: Dict[str, Any]) -> float:
        """
        Preserve original Stage 2 phi upper-bound logic.

        Original behavior:
            if self.phi100:
                phi_max = self.phi
            else:
                phi_max = min(200 * self.phi, user_phi_max)
                self.phi100 = True
        """
        if self.phi100:
            return float(self.phi)

        phi_max = min(200.0 * float(self.phi), float(phi_opts.get("phi_max", 1e4)))
        self.phi100 = True
        return float(phi_max)

    # ============================================================
    # Stage 1 warm start helpers
    # ============================================================

    def _stage1_residualize_covariates(
        self,
        y_donor: np.ndarray,
        donor_counts_safe: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 1 donor-level residual initialization.

        This keeps the original logic unchanged:
        - donor-level covariates Xd are used directly;
        - cell-level covariates Xc are averaged by donor;
        - statsmodels adds an intercept internally via sm.add_constant;
        - if the first Xc column is already an intercept, the OLS intercept
          is absorbed into alpha_init[0].
        """
        cov_parts: List[np.ndarray] = []

        if self.Xd is not None and self.pd > 0:
            if self.Xd.shape[0] != self.n:
                raise ValueError(f"Xd has inconsistent rows: {self.Xd.shape[0]} vs n={self.n}.")
            cov_parts.append(self.Xd.astype(np.float64))

        if self.Xc is not None and self.pc > 0:
            if self.Xc.shape[0] != self.M:
                raise ValueError(f"Xc has inconsistent rows: {self.Xc.shape[0]} vs M={self.M}.")

            Xc_means = np.zeros((self.n, self.pc), dtype=np.float64)
            for j in range(self.pc):
                s_j = _group_sum_numba(self.Xc[:, j].astype(np.float64), self.cell_donor, self.n)
                Xc_means[:, j] = s_j / donor_counts_safe
            cov_parts.append(Xc_means)

        alpha_init = np.zeros(self.s, dtype=np.float64)

        if len(cov_parts) == 0:
            y_resid = y_donor - float(np.mean(y_donor))
            return y_resid, alpha_init

        X_cov = np.concatenate(cov_parts, axis=1)
        X_cov = sm.add_constant(X_cov, has_constant="add")

        mask = np.isfinite(y_donor)
        mask &= np.all(np.isfinite(X_cov), axis=1)

        if int(mask.sum()) < max(3, X_cov.shape[1] + 1):
            y_resid = y_donor - float(np.nanmean(y_donor))
            return y_resid, alpha_init

        fit = sm.OLS(y_donor[mask], X_cov[mask]).fit()

        y_hat = np.empty_like(y_donor, dtype=np.float64)
        y_hat[:] = float(np.mean(y_donor[mask]))
        y_hat[mask] = fit.predict(X_cov[mask])
        y_resid = y_donor - y_hat

        coef = np.zeros(X_cov.shape[1], dtype=np.float64)
        coef[: fit.params.shape[0]] = fit.params

        intercept = float(coef[0])
        beta = coef[1:]

        offset = 0

        if self.Xd is not None and self.pd > 0:
            alpha_init[self.pc: self.pc + self.pd] = beta[offset: offset + self.pd]
            offset += self.pd

        if self.Xc is not None and self.pc > 0:
            alpha_init[0: self.pc] = beta[offset: offset + self.pc]
            offset += self.pc

        if self.Xc is not None and self.pc > 0:
            Xc0 = self.Xc[:, 0]
            if (
                np.all(np.isfinite(Xc0))
                and abs(np.mean(Xc0) - 1.0) < 1e-3
                and np.std(Xc0) < 1e-6
            ):
                alpha_init[0] += intercept

        return y_resid, alpha_init

    def _initialize_susie_from_residual(self, y_resid: np.ndarray) -> None:
        """
        Stage 1 SuSiE warm start from donor-level residual.
        """
        if self.p <= 0 or self.L <= 0:
            self.m_sum = np.zeros(self.p, dtype=np.float64)
            self.susie_q["prior_var"] = np.ones(self.L, dtype=np.float64)
            return

        susie = SuSiE(backend=self.backend)
        susie.fit(self.G, y_resid, L=self.L)

        pip_comp = np.asarray(susie.alpha, dtype=np.float64)
        post_mean = np.asarray(susie.mu, dtype=np.float64)
        prior_var = np.asarray(susie.V, dtype=np.float64).reshape(-1)
        post_mean2 = np.asarray(susie.mu2, dtype=np.float64)

        if pip_comp.shape != (self.L, self.p):
            raise ValueError(f"susie.alpha shape {pip_comp.shape} incompatible with ({self.L}, {self.p}).")
        if post_mean.shape != (self.L, self.p):
            raise ValueError(f"susie.mu shape {post_mean.shape} incompatible with ({self.L}, {self.p}).")
        if post_mean2.shape != (self.L, self.p):
            raise ValueError(f"susie.mu2 shape {post_mean2.shape} incompatible with ({self.L}, {self.p}).")
        if prior_var.shape != (self.L,):
            prior_var = np.ones(self.L, dtype=np.float64)

        pip_comp = np.clip(pip_comp, 0.0, 1.0)
        row_sum = np.maximum(pip_comp.sum(axis=1, keepdims=True), 1e-20)
        pip_comp = pip_comp / row_sum

        post_var = np.maximum(post_mean2 - post_mean ** 2, 1e-12)

        self.susie_q["pip_comp"] = pip_comp
        self.susie_q["post_mean"] = post_mean
        self.susie_q["post_var"] = post_var
        self.susie_q["prior_var"] = np.maximum(prior_var, 1e-12)
        self.susie_q["m_comp"] = pip_comp * post_mean
        self.m_sum = self.susie_q["m_comp"].sum(axis=0)

    def _update_tau_lower_bound_from_susie_prior(self) -> None:
        """
        Adaptive tau lower bound after Stage 1:

            tau_lower_bound = 0.01 * sum_l prior_var_l
        """
        prior_var_sum = float(np.sum(np.asarray(self.susie_q["prior_var"], dtype=np.float64)))

        if not np.isfinite(prior_var_sum) or prior_var_sum <= 0:
            prior_var_sum = 0.0

        self.tau_lower_bound = max(1e-12, 0.01 * prior_var_sum)

        if not self.tau_fixed:
            self.tau = float(max(float(self.tau), float(self.tau_lower_bound)))

    # ============================================================
    # Stage 1 warm start
    # ============================================================

    def _stage1_warm_start(self, options: Dict[str, Any], verbose: bool = True) -> None:
        phi_max = float(options.get("phi_max", 1e4))
        phi_min = float(options.get("phi_min", 1e-4))
        phi_init = float(options.get("phi_init", 1e4))
        estimate_phi = bool(options.get("estimate_phi", True))
        init_with_log = bool(options.get("init_with_log", False))

        y_sum = _group_sum_numba(self.y.astype(np.float64), self.cell_donor, self.n)
        donor_counts = _group_sum_numba(np.ones(self.M, dtype=np.float64), self.cell_donor, self.n)
        donor_counts_safe = np.maximum(donor_counts, 1.0)

        y_mean = y_sum / donor_counts_safe
        y_donor = np.log2(1 + y_sum) if init_with_log else y_mean

        y_resid, alpha_init = self._stage1_residualize_covariates(
            y_donor=y_donor,
            donor_counts_safe=donor_counts_safe,
        )

        self._initialize_susie_from_residual(y_resid)
        self._update_tau_lower_bound_from_susie_prior()

        if self.s > 0:
            self.alpha = alpha_init.astype(np.float64)

        if not np.isfinite(self.phi):
            if estimate_phi:
                self.C[:] = 0.0
                self._update_linear_predictor(include_genetics=True)

                self.phi = self._estimate_phi_mom(
                    phi_min=phi_min,
                    phi_max=phi_max,
                    phi_init=phi_init,
                )
            else:
                self.phi = float(np.clip(phi_init, phi_min, phi_max))

        self.history["phi"].append(float(self.phi))
        self.history["tau"].append(float(self.tau))
        self.history["tau_lower_bound"].append(float(self.tau_lower_bound))
        self.history["stage"].append("stage1")

        if verbose:
            response_name = "log1p(donor mean)" if init_with_log else "donor mean"
            tau_msg = f"tau_fixed={self.tau:.6g}" if self.tau_fixed else f"tau_init={self.tau:.6g}"
            print(
                f"[Stage 1] Warm start completed. "
                f"response={response_name}, phi={self.phi:.6g}, "
                f"{tau_msg}, tau_lower_bound={self.tau_lower_bound:.6g}."
            )


