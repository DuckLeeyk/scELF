# Input data and preprocessing

Prepare one expression vector and one candidate genotype matrix for each gene-cell-type fit. The model consumes numeric arrays, not an AnnData object, a VCF file, or a summary-statistics table.

## Array contract

Let `M` be the number of cells, `n` the number of donors, and `p` the number of candidate variants.

| Input | Shape | Meaning |
| --- | --- | --- |
| `y` | `(M,)` | Nonnegative integer expression counts for one gene |
| `Xc` | `(M, pc)` or `None` | Numeric cell-level covariates |
| `Xd` | `(n, pd)` or `None` | Numeric donor-level covariates |
| `Z` | `(M, n)` | One-hot cell-to-donor incidence matrix |
| `G` | `(n, p)` | Genotype dosages standardized across donors |

Keep `variant_ids` separately, in the same order as the columns of `G`. The model returns positional variant indices and does not store variant identifiers or genomic coordinates.

## Match cells and donors before fitting

Rows of `y`, `Xc`, and `Z` must describe the same cells in the same order. Columns of `Z`, rows of `G`, and rows of `Xd` must describe the same donors in the same order. Each row of `Z` must contain exactly one `1` and otherwise only `0`s. Retain donors with cells in the selected cell type.

The constructor converts arrays and derives donor assignments using `argmax(Z, axis=1)`. It does not verify identifier alignment or fully validate a one-hot matrix. Equal matrix dimensions alone do not establish alignment.

For your own data, explicitly reorder identifier-indexed tables before extracting arrays. In the following template, `cell_metadata` is already in expression-vector order, `genotype` has donor IDs as its index and variants as columns, and `donor_covariates` has donor IDs as its index:

```python
import numpy as np
import pandas as pd

cell_donors = cell_metadata["donor_id"].astype(str)
donor_ids = pd.Index(cell_donors.unique())

genotype = genotype.copy()
donor_covariates = donor_covariates.copy()
genotype.index = genotype.index.astype(str)
donor_covariates.index = donor_covariates.index.astype(str)
assert genotype.index.is_unique and donor_covariates.index.is_unique

# Missing donors raise KeyError; both tables now share the same donor order.
genotype = genotype.loc[donor_ids]
Xd = donor_covariates.loc[donor_ids].to_numpy(dtype=float)

cell_donor_index = donor_ids.get_indexer(cell_donors)
assert np.all(cell_donor_index >= 0)
Z = np.zeros((len(cell_donors), len(donor_ids)), dtype=np.uint8)
Z[np.arange(len(cell_donors)), cell_donor_index] = 1

G_raw = genotype.to_numpy(dtype=float)
assert np.isfinite(G_raw).all()
sd = G_raw.std(axis=0, ddof=0)
keep = sd > 0
assert keep.any()
variant_ids = genotype.columns.astype(str).to_numpy()[keep]
G = (G_raw[:, keep] - G_raw[:, keep].mean(axis=0)) / sd[keep]
```

This template allocates a dense `Z`, matching the current interface. Preserve a consistent donor-ID representation across all tables, especially identifiers with leading zeros.

## Expression and covariates

Supply counts on a nonnegative integer scale, such as UMI counts or deliberately constructed corrected counts. Do not substitute log-normalized expression, z-scores, or Pearson residuals for the count response. The model does not perform normalization or sequencing-depth correction automatically.

Covariates must already be numeric: dummy-encode categorical variables and exclude missing or infinite values. Add an intercept explicitly if desired, for example:

```python
Xc = np.column_stack([np.ones(y.size), cell_covariate_values])
```

There is no automatic intercept in the Stage 2 design. Avoid duplicate intercepts and redundant dummy columns across `Xc` and `Xd`. A library-size covariate has an estimated coefficient; the API does not expose a fixed log-library-size offset.

`Xc` and `Xd` can be `None`, but their arguments must still be supplied. To omit covariates from both stages, pass `Xc=None, Xd=None`. Setting `fit(use_covariates=False)` only changes the Stage 2 fixed-effect design; it does not remove covariates from Stage 1 initialization.

## Candidate variants

Select a cis-region and perform genotype quality control before fitting. Standardize each variant across donors using its donor-level mean and standard deviation. Remove monomorphic variants and handle missing genotypes upstream; no automatic imputation, minor-allele-frequency filtering, or region selection is implemented. Do not expand genotypes to cells and then standardize: that would weight donors by their cell counts.

## Preflight checks

After constructing all arrays, check their consistency before creating a model:

```python
assert y.ndim == 1 and np.isfinite(y).all()
assert np.all(y >= 0) and np.allclose(y, np.rint(y))
assert Z.shape == (y.size, G.shape[0])
assert np.all((Z == 0) | (Z == 1))
assert np.all(Z.sum(axis=1) == 1) and np.all(Z.sum(axis=0) > 0)
assert G.shape[1] > 0 and np.isfinite(G).all()
assert np.allclose(G.mean(axis=0), 0, atol=1e-8)
assert np.allclose(G.std(axis=0, ddof=0), 1, atol=1e-8)
for covariates, rows in ((Xc, y.size), (Xd, G.shape[0])):
    if covariates is not None:
        assert covariates.ndim == 2 and covariates.shape[0] == rows
        assert np.isfinite(covariates).all()
```

These checks supplement, rather than replace, identifier-based alignment. For production pipelines, use explicit exceptions rather than assertions if Python may run with optimization enabled.

## Relation to the manuscript preprocessing

The manuscript's real-data workflow uses sctransform-corrected counts followed by integer-preserving depth normalization, rather than Pearson residuals. Its cell-level covariates include log total UMI count, mitochondrial content, and expression PCs; donor-level covariates include age, sex, and genotype PCs, with pool indicators used to adjust for batch. The study also selects variants in the gene body plus 1 Mb on either side and applies an MAF threshold of 0.05.

These are study-specific preprocessing choices, not operations performed by `ScELF` or mandatory package defaults. Choose preprocessing and covariates appropriate to your dataset, while preserving the count response and array contract above. The [bundled example](example.md) starts from prepared input files.
