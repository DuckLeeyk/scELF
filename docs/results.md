# Results and interpretation

Call `.fit()` before reading posterior results. All variant arrays follow the column order of the standardized genotype matrix `G` supplied at construction.

## Extract the main outputs

Assuming the inputs are prepared as described in [Input data](input-data.md):

```python
import pandas as pd
from scelf import ScELF

model = ScELF(y=y, Xc=Xc, Xd=Xd, Z=Z, G=G, L=10, seed=42)
model.fit()

# Copy the credible-set-pruned PIPs stored by fit().
pip = model.pip.copy()
credible_sets = model.sets
variant_table = pd.DataFrame({"variant_id": variant_ids, "pip": pip})
variant_table.to_csv("variant_pips.csv", index=False)
```

`fit()` returns the model itself. Use a fresh instance for a new fit rather than treating repeated calls as an independent restart.

## Variant PIPs: pruned versus unpruned

A variant's PIP combines its component posterior probabilities as `1 - product(1 - component_probability)`. Components with prior variance at or below `prior_tol` are excluded.

| Access | Components included |
| --- | --- |
| `model.pip` immediately after `.fit()` | Active components with retained credible sets |
| `model.get_variant_pips(prune_by_cs=True, prior_tol=1e-9)` | Active components with retained credible sets |
| `model.get_variant_pips(prune_by_cs=False, prior_tol=1e-9)` | All active components, including those without retained credible sets |

The accessor defaults to `prune_by_cs=False`. It also updates `model.pip` in the normal calculation path, so calling it can replace the stored pruned result. Copy results when comparing definitions:

```python
pip_pruned = model.pip.copy()
pip_unpruned = model.get_variant_pips(prune_by_cs=False).copy()
```

Report which PIP definition you use. If no component survives credible-set filtering, pruned PIPs can all be zero even when unpruned component evidence exists. A PIP is not an association p-value and PIPs need not sum to one across variants when multiple effects are modeled.

## Credible sets

`model.sets` contains the credible sets calculated at the end of fitting. `model.get_credible_sets()` returns a newly calculated dictionary. The current implementation uses these fixed internal settings; they are not public fitting parameters:

| Rule | Current setting |
| --- | --- |
| Target component posterior coverage | `0.95` |
| Active component prior variance | Greater than `1e-9` |
| Minimum absolute genotype correlation for purity | `0.5` |
| Maximum variants sampled for purity calculation | `100` |
| Near-duplicate removal | Jaccard similarity at least `0.8` |

For each active component, variants are ordered by posterior weight and included until their cumulative weight reaches the target. Exact duplicate sets are removed, low-purity sets are filtered, and highly overlapping sets are reduced by retaining the stronger component according to the implemented ranking. Retained sets are ordered by minimum purity and then coverage.

| Dictionary field | Contents |
| --- | --- |
| `cs` | Mapping from component labels such as `L0` to zero-based variant-index arrays |
| `coverage` | Sum of the corresponding component's posterior weights within each set |
| `purity` | DataFrame containing `min_abs_corr`, `mean_abs_corr`, and `median_abs_corr` |
| `cs_index` | Original zero-based component indices used for pruned PIPs |
| `requested_coverage` | Target coverage, currently `0.95` |

Component labels retain their original indices; `L0` does not mean the strongest signal. Coverage is component posterior mass, not the sum of aggregate PIPs and not an observed empirical coverage rate. Purity uses absolute Pearson correlations of donor genotypes, not squared correlations. For sets larger than 100 variants, purity is estimated from a random subset.

When no set is retained, `cs` and `coverage` are `None`; `purity` and `cs_index` may be absent. Handle this case before iterating:

```python
for label, indices in (model.sets.get("cs") or {}).items():
    members = variant_ids[indices]
    coverage = float(model.sets["coverage"].loc[label])
    min_corr = float(model.sets["purity"].loc[label, "min_abs_corr"])
    print(label, members.tolist(), coverage, min_corr)
```

Repeated credible-set calculations can sample different purity subsets for large sets because they consume the model's random-number state. Keep the fitted `model.sets` when you want to reuse the same summary.

## Additional fitted quantities

| Attribute | Meaning |
| --- | --- |
| `model.phi` | Final negative-binomial size parameter |
| `model.tau` | Final donor random-effect variance |
| `model.alpha` | Fixed-effect coefficients, cell-level columns followed by donor-level columns |
| `model.C` | Fitted donor random effects in donor order |
| `model.m_sum` | Sum of component posterior mean variant effects, in `G` column order |
| `model.history` | Recorded objective and parameter trajectories |

`m_sum` is on the linear-predictor scale per standardized genotype unit. It is not an unstandardized dosage effect. The lower-level `susie_q` dictionary stores component posterior arrays; it is primarily useful for inspecting the implementation, whereas PIPs and credible sets are the main fine-mapping summaries.

No retained credible set does not by itself prove the absence of a genetic effect: assess the candidate region, data quality, fitting diagnostics, inactive components, and purity filtering. Conversely, a retained credible set expresses uncertainty under the fitted model and does not establish biological causality.
