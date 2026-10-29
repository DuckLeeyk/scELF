# scELF: A unified statistical framework for single-cell eQTL fine-mapping

scELF is a Python package for **single-cell-level cis-eQTL fine-mapping**. For one gene within one pre-specified cell type, it links expression counts from individual cells to donor genotypes, adjusts for cell-level and donor-level covariates, and accounts for dependence among cells from the same donor. It reports variant-level posterior inclusion probabilities (PIPs) and credible sets to summarize uncertainty in candidate causal variants.

## Framework overview

[![scELF framework: single-cell expression and donor genotypes, a negative-binomial mixed model with sparse genetic effects, and posterior fine-mapping outputs.](docs/assets/images/scelf-framework.png)](docs/assets/images/scelf-framework.png)

*Figure 1. Overview of the scELF framework. A cell-to-donor incidence matrix links cell-level counts and covariates to donor genotypes and covariates. A negative-binomial observation model combines fixed covariate effects, sum-of-single-effects genetic effects, and shared donor random effects. Posterior inference yields variant PIPs and credible sets. Click the figure to view the full-resolution image.*

The figure also illustrates downstream colocalization. This is separate from the core `ScELF.fit()` interface; the repository provides a dedicated PIP colocalization function and analysis workflows described below.

## Key features

- **Count-based inference:** a negative-binomial model accommodates overdispersed cell-level expression counts.
- **Within-donor dependence:** shared donor random effects account for repeated cells from the same individual.
- **Multiple genetic signals:** a sum-of-single-effects prior allows multiple candidate effects within a cis-region.
- **Covariate adjustment:** cell-level and donor-level covariates enter the fixed-effect design.
- **Posterior uncertainty:** variant PIPs and credible sets summarize fine-mapping evidence rather than association p-values.
- **Two-stage fitting:** a donor-level warm start initializes subsequent cell-level variational inference, with Numba and optional C++ computation backends.

Each fit analyzes one gene-cell-type pair. Cell annotation, candidate-region selection, genotype quality control, and expression preprocessing are performed upstream; the model does not automatically discover cell types or select eGenes.

## Installation

Python 3.10 or later is required. Clone the repository and install the package from its root directory, replacing `<repository-url>` with the repository URL:

```bash
git clone <repository-url> scELF
cd scELF
python -m pip install .
```

The distribution and import name are `scelf`; the public model class is `ScELF`. pip installs the declared runtime dependencies: NumPy, pandas, SciPy, statsmodels, and Numba.

For development, use an editable installation instead:

```bash
python -m pip install -e .
```

Verify the import:

```bash
python -c "from scelf import ScELF; print(ScELF)"
```

The default `backend="auto"` uses the C++ extension when a compatible build is available and otherwise falls back to Numba. Native compilation is optional for using scELF; the first Numba run can include compilation overhead. See the [installation guide](docs/installation.md) for details.

## Bundled B memory / LST1 example

The `example/` directory contains prepared counts, covariates, donor assignments, genotypes, and an end-to-end script for B memory cells in the LST1 cis-region. Run the smoke-test analysis from the repository root:

```bash
python example/run_b_memory_lst1.py
```

The default run performs **one Stage 2 iteration** to check the software and data flow. It is not a converged scientific analysis. To increase the iteration cap:

```bash
python example/run_b_memory_lst1.py --max-outer-iter 50
```

Convergence-based stopping remains enabled, so the actual iteration count may be lower than the cap. Inspect the fitting trajectory before interpreting results.

Cell-to-donor assignments are stored compactly in `cell_donor_index_B_memory.npy`: one zero-based donor index per cell, in expression-vector order. The script reconstructs the dense one-hot `Z` at runtime; the large dense text matrix is not needed.

The default output directory is `example/output/`:

| File | Contents |
| --- | --- |
| `B_memory_LST1_pip.csv` | Variant identifiers and credible-set-pruned PIPs |
| `B_memory_LST1_credible_sets.json` | Credible-set variant indices, coverage, and LD purity |
| `B_memory_LST1_performance.json` | Run metadata, backend, timing, and iteration information |

See the [example guide](docs/example.md) for input filenames and additional commands.

## Prepare your own inputs

For `M` cells, `n` donors, and `p` candidate cis variants, prepare the following numeric arrays:

| Argument | Shape | Description |
| --- | --- | --- |
| `y` | `(M,)` | Nonnegative integer expression counts for one gene |
| `Xc` | `(M, pc)` or `None` | Cell-level covariates |
| `Xd` | `(n, pd)` or `None` | Donor-level covariates |
| `Z` | `(M, n)` | One-hot cell-to-donor incidence matrix |
| `G` | `(n, p)` | Genotype dosages standardized across donors |

Align `y`, `Xc`, and the rows of `Z` by cell order. Align `G`, `Xd`, and the columns of `Z` by donor order. Keep variant identifiers separately in `G` column order.

Standardize each genotype column across donors and remove zero-variance variants before fitting; scELF does not standardize or impute genotypes automatically. Do not use log expression or Pearson residuals as the count response. Supply finite numeric covariates and add an intercept explicitly if desired. See [Input data and preprocessing](docs/input-data.md) for alignment templates and validation checks.

## Minimal Python usage

With the arrays prepared as above:

```python
from scelf import ScELF

model = ScELF(y=y, Xc=Xc, Xd=Xd, Z=Z, G=G, L=10, seed=42)
model.fit()

pip = model.pip.copy()
credible_sets = model.sets
```

`L` sets the number of single-effect components, not an assumed exact number of causal variants. `.fit()` uses the class defaults, including a maximum of 50 Stage 2 iterations; the example script has its own overrides.

Immediately after fitting, `model.pip` contains credible-set-pruned PIPs. In contrast, `model.get_variant_pips()` defaults to unpruned PIPs and can replace the stored result. Use `prune_by_cs=True` when requesting the pruned definition explicitly. Credible-set indices are zero-based positions in `G` column order. See [Results and interpretation](docs/results.md) for details.

## Documentation

The documentation source is maintained in `docs/`:

- [Overview](docs/index.md) and [model and inference](docs/model.md)
- [Installation](docs/installation.md)
- [Input data and preprocessing](docs/input-data.md)
- [B memory / LST1 example](docs/example.md)
- [Results and interpretation](docs/results.md)
- [Fitting diagnostics](docs/diagnostics.md)
- [Parameter reference](docs/parameters.md), including defaults and nested fitting options

PIPs and credible sets are approximate posterior summaries conditional on the supplied data, candidate variants, and model assumptions; they do not establish biological causality.

## Repository organization

```text
src/scelf/          # Installable model package and numerical backends
example/           # Prepared B memory / LST1 inputs and runnable script
docs/              # Documentation pages and framework figure
analysis/onek1k/   # Manuscript analysis workflows, separate from the package
tests/             # Implementation checks
pyproject.toml     # Package metadata and dependency declarations
setup.py           # Optional native-extension build configuration
```

## PIP colocalization and manuscript workflows

The package also exports `coloc_pip` for conditional H3/H4 scoring of aligned PIP
vectors. See the [colocalization API guide](docs/colocalization.md).

The [OneK1K analysis workflows](analysis/onek1k/README.md) provide argparse entry
points for input preparation, four-method fine-mapping, GWAS colocalization,
TWAS and locus tables. Plotting and external validation scripts are excluded.

## Development checks

Install the test extra and run the suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

GitHub Actions checks package installation and synthetic fits, Numba-only fallback,
required C++ builds, compact example loading, optional analysis workflows, and a
strict documentation build. Native or optional-analysis tests skip locally when
their dependencies are absent; historical baseline comparisons require a
user-supplied reference snapshot. CI does not claim to reproduce study-wide
results or execute the complete SCT/SAIGE-QTL R pipelines.

## License

The scELF software is licensed under the [GNU General Public License v3.0](LICENSE)
(SPDX: `GPL-3.0-only`). Third-party dependencies retain their own licenses. The
software license does not replace the access or redistribution terms of upstream
research datasets.
