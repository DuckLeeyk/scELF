# B memory / LST1 example

The repository includes prepared input data and an end-to-end script for fine-mapping the LST1 cis-region in B memory cells. Run commands below from the repository root after [installing scELF](installation.md).

## Included files

```text
example/
├── GReX_B_memory_LST1.txt             # cell-level expression counts
├── cell_covariates_B memory.csv       # cell-level covariates
├── cell_donor_index_B_memory.npy      # zero-based donor index per cell
├── genotype_IDs_B_memory.txt          # donor IDs in matrix order
├── individual_covariates_dummy.csv    # donor-level covariates
├── genotype_B_memory_LST1.csv         # candidate genotypes
└── run_b_memory_lst1.py               # analysis script
```

The filename prefix `GReX` does not change the response contract: the expression vector is used as a count response, not as log expression or residuals.

The mapping file is a one-dimensional NumPy integer array, in expression-vector cell order. Each value is a zero-based row index into `genotype_IDs_B_memory.txt`. The loader reconstructs the dense one-hot `Z` at runtime, preserving the model interface while avoiding a large text matrix of zeros. Load it with `np.load(path, allow_pickle=False)`; it contains no Python objects.

## Run a quick check

```bash
python example/run_b_memory_lst1.py
```

The script loads the prepared files, checks dimensions and finite values, aligns donor covariates by donor ID, standardizes genotypes across donors, and fits `ScELF` with `L=10` and seed 42 by default.

!!! warning "One iteration is not a converged fit"

    The default is one Stage 2 iteration, intended to check software and data flow. Do not treat its output as a final fine-mapping analysis.

The script trusts that genotype rows already follow the donor-ID file and that expression, cell covariates, and `Z` share cell order. It checks genotype row count, but does not reorder genotype rows by their identifiers. For new data, explicitly align all IDs as described in [Input data](input-data.md).

## Run a longer fit

```bash
python example/run_b_memory_lst1.py --max-outer-iter 50
```

This increases the Stage 2 iteration cap, while keeping convergence-based stopping enabled. Review the trajectory and [stopping rule](diagnostics.md) before interpreting results. The script uses `tol_F=1e-5` and a Stage 1 `phi_init=100` fallback, whereas the class defaults are `1e-6` and `1e4`, respectively.

To select a backend explicitly:

```bash
python example/run_b_memory_lst1.py --backend numba --max-outer-iter 50
```

`--backend cpp` requires a compatible compiled extension. Omit the option to use automatic selection. The script prints the source path and actual backend; when launched from a checkout it prefers that checkout's `src/scelf` over an older installed package.

## Output files

The default output directory is `example/output/`.

| File | Contents |
| --- | --- |
| `B_memory_LST1_pip.csv` | Variant identifiers and credible-set-pruned PIPs |
| `B_memory_LST1_credible_sets.json` | Credible-set indices, component coverage, and LD purity summaries |
| `B_memory_LST1_performance.json` | Run metadata, backend, timing, and iteration information |

Credible-set indices are zero-based positions in the input genotype columns. Use the variant order in the PIP table to recover their identifiers. See [Results and interpretation](results.md) for the filtering rules and empty-set behavior.

Write results elsewhere:

```bash
python example/run_b_memory_lst1.py --max-outer-iter 50 --output-dir path/to/results
```

## Minimal Python interface

With prepared arrays `y`, `Xc`, `Xd`, `Z`, and `G`:

```python
from scelf import ScELF

model = ScELF(y=y, Xc=Xc, Xd=Xd, Z=Z, G=G, L=10, seed=42)
model.fit()

pip = model.pip.copy()
credible_sets = model.sets
```

This minimal call uses the class defaults, not all script-specific overrides. See [Parameter reference](parameters.md) for every exposed fitting option.
