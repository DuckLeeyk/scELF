# Model and inference

scELF analyzes one gene within one pre-specified cell type, using expression counts from multiple cells per donor and genotypes from the corresponding donors. The final fine-mapping likelihood is defined at the cell level; donor aggregation is used only for initialization.

## Negative-binomial observation model

For cell `k` from donor `i`, the model uses the following mean-size parameterization:

```text
y_ik ~ NegativeBinomial(mu_ik, phi)
mu_ik = exp(eta_ik)
Var(y_ik | eta_ik, phi) = mu_ik + mu_ik^2 / phi

eta_ik = x_c,ik^T alpha_c + x_d,i^T alpha_d + g_i^T b + c_i
c_i ~ Normal(0, tau)
```

`phi` is a positive size parameter shared across cells in a fit. Smaller `phi` allows greater overdispersion; large `phi` approaches a Poisson model. `tau` is the variance of the donor random effect, not its standard deviation. Cells from the same donor share `c_i`, while measured cell-level and donor-level covariates enter as fixed effects.

In matrix notation, `Z` maps donors to cells, the fixed-effect design is `[Xc, Z @ Xd]`, and the genetic design is `Z @ G`. Genotypes must be standardized across donors, not across repeated cells. See [Input data](input-data.md) for the exact array interface.

## Sum-of-single-effects prior

The total variant-effect vector is a sum of `L` single-effect components:

```text
b = b_1 + ... + b_L
b_l = beta_l * gamma_l
beta_l ~ Normal(0, prior_var_l)
gamma_l selects one of p variants, with prior probability 1 / p
```

Each component represents one candidate genetic effect. Their sum permits multiple signals in the same cis-region. `L` sets the component capacity; it is not a claim that exactly `L` causal variants exist. Component prior variances can shrink towards zero during fitting, and the result code excludes inactive components.

The implementation uses uniform variant prior probabilities. The public `ScELF` constructor does not expose annotation-dependent variant priors.

## Two-stage inference

### Stage 1: donor-level warm start

The default initialization averages expression counts within each donor, residualizes donor-level and aggregated cell-level covariates, and fits a Gaussian SuSiE warm start. This initializes component posteriors and fixed effects. If `phi` is not supplied, the initial size parameter is estimated by method of moments unless disabled in `stage1_options`.

The optional `init_with_log=True` response is `log2(1 + donor total counts)`. It changes the warm start only, not the cell-level response used in Stage 2.

### Stage 2: cell-level variational fitting

The fitting loop uses a Jaakkola-Jordan quadratic bound for the negative-binomial likelihood and a Gaussian working representation. Donor random effects are integrated out for genetic posterior updates, and a restricted maximum-likelihood working criterion adjusts for fixed effects.

Each outer iteration alternates working-quantity updates, iterative Bayesian stepwise selection (IBSS) of the single-effect posteriors, and updates of fixed effects, donor effects, and enabled variance/size parameters. By default, each iteration performs one IBSS sweep, updates component prior variances, estimates `phi` by method of moments, and updates `tau` using bounded scalar optimization.

The local bound parameter uses the current fitted predictor, `xi = abs(eta - log(phi))`. The recorded objective is a working variational criterion; with plug-in bound and size updates, monotone ascent should not be assumed. See [Fitting diagnostics](diagnostics.md) for the actual stopping rule.

### Posterior summaries

After fitting, scELF constructs credible sets and aggregates component probabilities into variant PIPs. The stored `model.pip` uses only components whose credible sets survive filtering. This differs from the default unpruned accessor call; see [Results and interpretation](results.md).

## Implementation map

The public entry point is `from scelf import ScELF`. The package separates the implementation into focused modules:

| Module in `src/scelf/` | Responsibility |
| --- | --- |
| `model.py` | Public class, Stage 2 loop, posterior and parameter updates |
| `stage1.py` | Donor-level warm start and initialization helpers |
| `susie.py` | Gaussian SuSiE implementation used for initialization |
| `results.py` | PIP aggregation, credible-set construction, and LD purity |
| `solver.py` | Exact Woodbury-based working-covariance solver |
| `numerics.py` | Numerical helpers and compiled aggregation operations |
| `backend.py` and `native/` | Backend selection and optional C++ kernels |

Donor-level sufficient statistics and the Woodbury structure avoid constructing a dense cell-by-cell covariance matrix. However, the public interface still accepts and stores a dense cell-to-donor matrix `Z`; do not assume sparse-matrix support or memory use independent of the number of cells.

## Scope and assumptions

The model expects prepared count-like expression data, finite numeric covariates, and standardized candidate genotypes. It does not perform cell clustering, genotype imputation, cis-window selection, or genome-wide association discovery. PIPs and credible sets are approximate posterior summaries conditional on the supplied candidate variants, covariates, model assumptions, and fitted hyperparameters, not experimental proof of causality.
