# Fitting diagnostics

The default example is a smoke check, not a convergence assessment. Before interpreting results, inspect the fitting trajectory and verify that the inputs satisfy the [data contract](input-data.md).

## Inspect the recorded history

For a fresh model fitted once, `history["F"]` contains one objective value per Stage 2 iteration. The parameter and stage histories also include a Stage 1 initialization entry, so they are one element longer. Align the records explicitly:

```python
import numpy as np
import pandas as pd

history = model.history
trajectory = pd.DataFrame({
    "iteration": np.arange(1, len(history["F"]) + 1),
    "F": history["F"],
    "phi": history["phi"][1:],
    "tau": history["tau"][1:],
    "tau_lower_bound": history["tau_lower_bound"][1:],
})
print(trajectory.tail())
print("Finite trajectory:", np.isfinite(trajectory.to_numpy()).all())
```

The code assumes one call to `.fit()` on a fresh instance. Reusing an instance can retain history and update flags from an earlier fit.

## Understand the stopping rule

After two Stage 2 objective values are available, fitting stops when:

```text
F_new - F_previous <= tol_F * max(1, abs(F_previous))
```

This is an increase-based rule, not an absolute-change rule: an objective decrease also satisfies it. The iteration count is additionally capped by `max_outer_iter`, or by `stage2_options["max_iter"]` if supplied. The corresponding nested `tol_F` also overrides the top-level value.

The objective is a working variational criterion with plug-in local-bound and method-of-moments size updates; monotonicity is not guaranteed. Completion of `.fit()` is not a convergence certificate, and there is no public `converged_` attribute. Inspect verbose output, objective differences, and parameter behavior rather than using the iteration count alone.

For a longer example run:

```bash
python example/run_b_memory_lst1.py --max-outer-iter 50
```

This raises the iteration cap; it does not guarantee 50 iterations or convergence. If the cap is reached with continuing changes, rerun a fresh model with a larger cap and compare the outputs. If an early objective decrease triggers stopping, inspect the inputs and parameter trajectory before interpreting the fit.

## Parameter boundaries

The adaptive lower bound for `tau` is initialized after Stage 1 as `max(1e-12, 0.01 * sum(component_prior_variances))`. When `tau` is estimated, its effective update bound also includes the user-specified `tau_min`. Once an update reaches the lower boundary, the implementation freezes `tau` there. Check `model.tau_frozen_at_lower_bound` alongside the recorded trajectory; a boundary value is not an unconstrained estimate of exactly zero variance.

The first Stage 2 `phi` update uses an upper bound of `min(200 * current_phi, user_phi_max)`. Later updates use the current `phi` as their upper bound. Consequently, changing `phi_max` does not make all later updates unconstrained. Supplying `phi` to the constructor initializes it; to prevent Stage 2 updates, also set:

```python
model.fit(stage2_options={"phi": {"update": False}})
```

See [Parameter reference](parameters.md) for the complete defaults and controls.

## Common issues

| Observation | Checks and interpretation |
| --- | --- |
| Dimension errors or implausible effects | Verify cell and donor IDs and their order, not only matrix sizes. |
| Invalid or unstable numerical values | Check finite inputs, integer nonnegative counts, genotype variation, and redundant covariate columns. |
| All stored PIPs are zero | Inspect retained credible sets and compare pruned versus unpruned PIPs; purity filtering can remove components. |
| No credible set is retained | Inspect candidate variants, LD purity, component shrinkage, and fitting stability; this is not automatically a software error. |
| An option appears to have no effect | Check dictionary spelling and nesting. Unknown keys are not comprehensively rejected; the former `rho` control is not a supported fitting option. |
| First run is slower | Numba may compile kernels on first use. Check `backend_info()` and `model.backend` to identify the actual backend. |

Use the `seed` argument for stochastic steps such as purity subsampling and optional stochastic trace estimation. Numerical libraries and backend differences can still affect floating-point results. For independent gene-cell-type fits, use independent model instances and retain the input/variant ordering with each result.
