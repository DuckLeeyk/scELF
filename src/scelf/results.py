"""PIP and credible-set accessors for scelf.ScELF."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

class _ResultsMixin:
    # ============================================================
    # Public accessors
    # ============================================================

    def get_variant_pips(
        self,
        prune_by_cs: bool = False,
        prior_tol: float = 1e-9,
    ) -> np.ndarray:
        if "pip_comp" not in self.susie_q:
            pip = np.zeros(self.p, dtype=np.float64)
            self.pip = pip
            return pip

        gamma = np.asarray(self.susie_q["pip_comp"], dtype=np.float64)

        if gamma.shape != (self.L, self.p):
            raise ValueError(f"pip_comp has shape {gamma.shape}, expected ({self.L}, {self.p}).")

        prior_var = np.asarray(self.susie_q.get("prior_var", np.ones(self.L)), dtype=np.float64)
        if prior_var.shape[0] != self.L:
            prior_var = np.ones(self.L, dtype=np.float64)

        include_idx = np.where(prior_var > float(prior_tol))[0]

        if prune_by_cs:
            if not isinstance(self.sets, dict) or self.sets.get("cs_index", None) is None:
                self.sets = self.get_credible_sets()

            cs_idx = self.sets.get("cs_index", None)
            if cs_idx is None:
                include_idx = np.array([], dtype=int)
            else:
                include_idx = np.intersect1d(include_idx, np.asarray(cs_idx, dtype=int))

        if include_idx.size == 0:
            return np.zeros(self.p, dtype=np.float64)

        res = gamma[include_idx, :]
        pip = 1.0 - np.prod(1.0 - np.clip(res, 0.0, 1.0), axis=0)
        pip = np.clip(pip, 0.0, 1.0)

        self.pip = pip
        return pip

    def get_credible_sets(self) -> Dict[str, Any]:
        if self.p == 0 or self.L == 0:
            return {"cs": None, "coverage": None, "requested_coverage": 0.95}

        if "pip_comp" not in self.susie_q:
            return {"cs": None, "coverage": None, "requested_coverage": 0.95}

        weights_matrix = np.asarray(self.susie_q["pip_comp"], dtype=np.float64)

        if weights_matrix.shape != (self.L, self.p):
            raise ValueError(f"pip_comp has shape {weights_matrix.shape}, expected ({self.L}, {self.p}).")

        post_mean = np.asarray(
            self.susie_q.get("post_mean", np.zeros((self.L, self.p))),
            dtype=np.float64,
        )

        if post_mean.shape != (self.L, self.p):
            post_mean = np.zeros((self.L, self.p), dtype=np.float64)

        G = np.asarray(self.G, dtype=np.float64)

        prior_var = np.asarray(self.susie_q.get("prior_var", np.ones(self.L)), dtype=np.float64)
        if prior_var.shape[0] != self.L:
            prior_var = np.ones(self.L, dtype=np.float64)

        coverage = 0.95
        min_abs_corr = 0.5
        n_purity = 100

        def _row_cs_indices(alpha_row: np.ndarray, mu_row: np.ndarray, cov: float) -> np.ndarray:
            x = np.asarray(alpha_row, dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.clip(x, 0.0, 1.0)

            abs_mu = np.abs(np.asarray(mu_row, dtype=np.float64))
            abs_mu = np.nan_to_num(abs_mu, nan=0.0, posinf=0.0, neginf=0.0)

            idx = np.arange(x.shape[0], dtype=np.int64)
            order = np.lexsort((idx, -abs_mu, -x))

            x_sorted = x[order]
            csum = np.cumsum(x_sorted)

            k = int(np.searchsorted(csum, cov, side="left")) + 1
            k = min(k, x.shape[0])

            return order[:k].astype(np.int64)

        def _purity_from_G(pos: np.ndarray, n0: int = 100) -> List[float]:
            pos = np.asarray(pos, dtype=int)

            if pos.size <= 1:
                return [1.0, 1.0, 1.0]

            if pos.size > n0:
                pos = self.rng.choice(pos, size=n0, replace=False)

            if G.shape[0] <= 1:
                sub = np.eye(pos.size, dtype=np.float64)
            else:
                selected = G[:, pos]
                Gc = selected - np.nanmean(selected, axis=0, keepdims=True)
                std = np.sqrt(np.maximum(np.nanvar(Gc, axis=0, ddof=1), 1e-12))
                Gz = Gc / std
                R = (Gz.T @ Gz) / max(G.shape[0] - 1, 1)
                np.clip(R, -1.0, 1.0, out=R)
                sub = np.abs(0.5 * (R + R.T))

            if sub.shape[0] <= 1:
                return [1.0, 1.0, 1.0]

            tri = sub[np.triu_indices(sub.shape[0], k=1)]

            if tri.size == 0:
                return [1.0, 1.0, 1.0]

            return [float(np.min(tri)), float(np.mean(tri)), float(np.median(tri))]

        cs_list = []
        claimed_coverage = np.zeros(self.L, dtype=np.float64)

        for l in range(self.L):
            cs = _row_cs_indices(weights_matrix[l, :], post_mean[l, :], coverage)
            cs_list.append(cs)
            claimed_coverage[l] = float(weights_matrix[l, cs].sum()) if cs.size > 0 else 0.0

        include_idx = (
            (prior_var > 1e-9)
            & np.array([cs_list[l].size > 0 for l in range(self.L)], dtype=bool)
        )

        if not np.any(include_idx):
            return {"cs": None, "coverage": None, "requested_coverage": float(coverage)}

        first_seen: Dict[tuple, int] = {}
        unique_mask = np.zeros(self.L, dtype=bool)

        for l in range(self.L):
            if not include_idx[l]:
                continue

            key = tuple(np.sort(cs_list[l]).tolist())

            if key not in first_seen:
                first_seen[key] = l
                unique_mask[l] = True

        include_idx = include_idx & unique_mask

        if not np.any(include_idx):
            return {"cs": None, "coverage": None, "requested_coverage": float(coverage)}

        kept_idx = np.where(include_idx)[0]
        cs_kept = [cs_list[i] for i in kept_idx]
        cov_kept = claimed_coverage[kept_idx]

        purity_rows = [_purity_from_G(cs, n0=n_purity) for cs in cs_kept]
        purity_df = pd.DataFrame(
            purity_rows,
            columns=["min_abs_corr", "mean_abs_corr", "median_abs_corr"],
        )

        ok = np.where(purity_df["min_abs_corr"].values >= min_abs_corr)[0]

        if ok.size == 0:
            return {"cs": None, "coverage": None, "requested_coverage": float(coverage)}

        cs_kept = [cs_kept[i] for i in ok]
        cov_kept = cov_kept[ok]
        kept_idx = kept_idx[ok]
        purity_df = purity_df.iloc[ok, :].reset_index(drop=True)

        jaccard_thresh = 0.8

        def _comp_score(orig_l: int, cs: np.ndarray) -> tuple:
            if cs.size == 0:
                return (0.0, 0.0, -10**9)

            w = np.asarray(weights_matrix[orig_l, cs]).ravel()
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

            return (float(np.max(w)), float(np.sum(w)), -int(cs.size))

        scores = [_comp_score(int(kept_idx[i]), cs_kept[i]) for i in range(len(kept_idx))]
        order = sorted(range(len(kept_idx)), key=scores.__getitem__, reverse=True)

        kept_flags = np.zeros(len(kept_idx), dtype=bool)
        kept_sets: List[set] = []

        for ii in order:
            s_new = set(map(int, cs_kept[ii].tolist()))

            if len(s_new) == 0:
                continue

            too_similar = False

            for s_old in kept_sets:
                inter = len(s_new & s_old)
                union = len(s_new | s_old)
                jac = inter / union if union > 0 else 0.0

                if jac >= jaccard_thresh:
                    too_similar = True
                    break

            if not too_similar:
                kept_flags[ii] = True
                kept_sets.append(s_new)

        if not np.any(kept_flags):
            return {"cs": None, "coverage": None, "requested_coverage": float(coverage)}

        cs_kept = [cs_kept[i] for i in range(len(cs_kept)) if kept_flags[i]]
        cov_kept = cov_kept[kept_flags]
        kept_idx = kept_idx[kept_flags]
        purity_df = purity_df.iloc[np.where(kept_flags)[0], :].reset_index(drop=True)

        purity_vals = purity_df["min_abs_corr"].values
        order2 = np.lexsort((-cov_kept, -purity_vals))

        cs_kept = [cs_kept[i] for i in order2]
        cov_kept = cov_kept[order2]
        kept_idx = kept_idx[order2]
        purity_df = purity_df.iloc[order2, :].reset_index(drop=True)

        row_names = [f"L{int(i)}" for i in kept_idx]

        cs_dict = dict(zip(row_names, cs_kept))
        purity_df.index = row_names

        return {
            "cs": cs_dict,
            "purity": purity_df,
            "cs_index": kept_idx,
            "coverage": pd.Series(cov_kept, index=row_names),
            "requested_coverage": float(coverage),
        }


