# Parameter reference

All default values below are taken from the current `scelf` source code. `stage1_options` and `stage2_options` are optional dictionaries; omitting them applies the documented defaults.

## `ScELF(...)`

```python
ScELF(
    y, Xc, Xd, Z, G, L,
    phi=None, seed=None, tau=None, tau_fixed=None, backend="auto",
)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `y` | Required | Nonnegative integer cell-level expression counts with length `M`. |
| `Xc` | Required; may be `None` | Cell-level covariate matrix with shape `(M, pc)`. |
| `Xd` | Required; may be `None` | Donor-level covariate matrix with shape `(n, pd)`. |
| `Z` | Required | Cell-to-donor incidence matrix with shape `(M, n)`; every cell maps to one donor. |
| `G` | Required | Finite donor-level genotype matrix with shape `(n, p)`, standardized across donors. |
| `L` | Required | Positive number of SuSiE single-effect components; a capacity, not a fixed causal-variant count. |
| `phi` | `None` | Positive initial negative-binomial size parameter. Estimated in Stage 1 when `None`; supplying it does not disable Stage 2 updates. |
| `seed` | `None` | Random seed for the model's random-number generator, including purity subsampling and stochastic trace estimation when used. |
| `tau` | `None` | Donor random-effect variance. When `None`, it is initialized to `0.01`. |
| `tau_fixed` | `None` | Whether to hold `tau` fixed. With `None`, `tau` is fixed when it is supplied and otherwise updated in Stage 2. |
| `backend` | `"auto"` | `"auto"` uses C++ when available, otherwise Numba. `"numba"` forces Numba; `"cpp"` requires a compatible extension and raises an error if unavailable. |

`tau` is a variance, not a standard deviation. With `tau_fixed=True` and `tau=None`, the fixed value is `0.01`. To supply an initial `tau` but continue estimating it, set `tau_fixed=False` explicitly. To hold a supplied `phi` fixed in Stage 2, set `stage2_options={"phi": {"update": False}}` in `.fit()`.

!!! warning "Genotype preprocessing"

    `ScELF` does not standardize `G` automatically. Center and scale each variant before constructing the model, as in the bundled example, and remove zero-variance variants.

See [Input data](input-data.md) for identifier alignment, explicit intercepts, and the complete array contract. Sparse matrix inputs are not supported by the documented interface.

## `.fit(...)`

```python
model.fit(
    use_covariates=True,
    max_outer_iter=50,
    tol_F=1e-6,
    verbose=True,
    stage1_options=None,
    stage2_options=None,
)
```

| Parameter | Default | Description |
| --- | --- | --- |
| `use_covariates` | `True` | Whether to include `Xc` and `Xd` during Stage 2. Stage 1 still uses supplied covariates; pass both as `None` to omit them from both stages. |
| `max_outer_iter` | `50` | Maximum number of Stage 2 variational-EM iterations. |
| `tol_F` | `1e-6` | Relative working-objective-increase threshold used to stop Stage 2; decreases also trigger stopping. See the exact rule in [Fitting diagnostics](diagnostics.md). |
| `verbose` | `True` | Whether to print fitting progress to standard output. |
| `stage1_options` | `None` | Stage 1 warm-start configuration dictionary. |
| `stage2_options` | `None` | Stage 2 variational-EM configuration dictionary. |

If provided, `stage2_options["max_iter"]` and `stage2_options["tol_F"]` override `.fit()` values for `max_outer_iter` and `tol_F`, respectively.

`.fit()` returns the fitted `ScELF` instance. Use a fresh instance for each independent fit. Option dictionaries are read by key, and unknown keys are not comprehensively validated; check spelling and nesting carefully.

## `stage1_options`

```python
stage1_options = {
    "estimate_phi": True,
    "phi_min": 1e-4,
    "phi_max": 1e4,
    "phi_init": 1e4,
    "init_with_log": False,
}
```

| Key | Default | Description |
| --- | --- | --- |
| `estimate_phi` | `True` | Whether to estimate the initial `phi` by method of moments. |
| `phi_min` | `1e-4` | Lower bound for Stage 1 `phi`. |
| `phi_max` | `1e4` | Upper bound for Stage 1 `phi`. |
| `phi_init` | `1e4` | Fallback initial value when phi estimation fails or is disabled. |
| `init_with_log` | `False` | Whether to use `log2(1 + donor total counts)` rather than donor mean counts for warm start. |

## `stage2_options`

```python
stage2_options = {
    "max_iter": 50,
    "tol_F": 1e-6,
    "tau": {...},
    "phi": {...},
    "e_step": {...},
}
```

| Key | Default | Description |
| --- | --- | --- |
| `max_iter` | `.fit(max_outer_iter)` | Overrides the Stage 2 maximum iteration count. |
| `tol_F` | `.fit(tol_F)` | Overrides the Stage 2 working-objective-increase stopping threshold. |
| `tau` | `{}` | Tau-update configuration; see below. |
| `phi` | `{}` | Stage 2 phi-update configuration; see below. |
| `e_step` | `{}` | IBSS E-step configuration; see below. |

### `stage2_options["tau"]`

| Key | Default | Description |
| --- | --- | --- |
| `method` | `"bounded"` | Tau-update method. `"bounded"`, `"scalar"`, and `"brent"` use bounded scalar optimization; `"ai"`, `"aireml"`, and `"ai_reml"` use a complete-data-information score update. |
| `tau_min` | `1e-8` | User-supplied tau lower bound. The effective bound is `max(tau_min, adaptive tau_lower_bound, 1e-12)`. |
| `tau_max` | `1e8` | Tau upper bound. The code ensures it is at least ten times the effective lower bound. |
| `xatol` | `1e-5` | Absolute tolerance for bounded scalar optimization methods only. |
| `boundary_tol` | Computed | For bounded methods only; defaults to `max(1e-10, xatol * max(1, effective_tau_min))`. Tau is frozen when it reaches the lower-bound boundary. |
| `ai_max_iter` | `5` | Maximum number of internal score updates for AI methods only. |
| `trace_samples` | `16` | Number of Hutchinson trace samples for AI methods only. |

The adaptive `tau_lower_bound` is initialized from Stage 1 component prior variances as `max(1e-12, 0.01 * sum(prior_var))`. Tau-update settings have no effect when `tau_fixed=True`. The AI aliases use the implementation's complete-data-information score update with stochastic traces; they should not be interpreted as a conventional average-information REML implementation.

### `stage2_options["phi"]`

| Key | Default | Description |
| --- | --- | --- |
| `update` | `True` | Whether to update `phi` during Stage 2. |
| `phi_min` | `1e-4` | Lower bound for Stage 2 `phi`. |
| `phi_max` | `1e4` | User upper bound for the first Stage 2 update. Its effective first upper bound is `min(200 × current_phi, phi_max)`; subsequent updates use the current phi as their upper bound. |
| `phi_init` | `1e4` | Fallback value if the phi method-of-moments update fails. |

### `stage2_options["e_step"]`

| Key | Default | Description |
| --- | --- | --- |
| `ibss_sweeps` | `1` | Number of IBSS sweeps in every Stage 2 outer iteration; the implementation enforces at least one. |
| `update_prior_var` | `True` | Whether to update each SuSiE component's prior variance. |

## Result accessors

```python
model.get_variant_pips(prune_by_cs=False, prior_tol=1e-9)
model.get_credible_sets()
```

| Argument | Default | Description |
| --- | --- | --- |
| `prune_by_cs` | `False` | Restrict PIP aggregation to components with retained credible sets when `True`. |
| `prior_tol` | `1e-9` | Include only components with prior variance greater than this value in PIP aggregation. |

`get_credible_sets()` takes no arguments. Its current fixed settings are 95% target component coverage, minimum absolute correlation 0.5, at most 100 variants for purity calculation, and Jaccard similarity 0.8 for near-duplicate filtering. These are implementation settings, not exposed fitting options. See [Results and interpretation](results.md) for field types and variant indexing.

## Fitted results

| Attribute or method | Description |
| --- | --- |
| `model.pip` | Credible-set-pruned variant PIPs immediately after `.fit()`. A later accessor call can replace them. |
| `model.sets` | Credible sets, coverage, purity, and component indices. |
| `model.history` | Working objective (`F`), `phi`, `tau`, `tau_lower_bound`, and stage records. Parameter histories include Stage 1; `F` includes Stage 2 only. |
| `model.get_variant_pips()` | Returns unpruned PIPs by default; use `prune_by_cs=True` for the fitting-output definition. |
| `model.get_credible_sets()` | Returns a credible-set results dictionary. |
