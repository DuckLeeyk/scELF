# Numba and exact donor compression

scELF uses donor-level sufficient statistics to reduce repeated cell-level work while retaining the cell-level count likelihood. The implementation stores genotypes in float64 Fortran order, performs compiled aggregation, and uses an exact Woodbury-based solver. The dense input `Z` is still accepted and stored.

## Exact donor identity

Let `w = Z.T @ W`, `c = 1 / (1 + tau*w)`, `u = w*c`, `T = Z.T @ V @ X`, and `A = X.T @ V @ X + 1e-10*I`. The donor-level projection satisfies:

```text
H = Z.T @ P @ Z = diag(u) - T @ inv(A) @ T.T
b = Z.T @ P @ y_work
a_l = G.T @ (b - H @ (G @ m_minus_l))
```

`H` is applied as a diagonal plus low-rank operator; neither `H` nor a variant-by-variant Gram matrix is materialized. Component updates remain sequential.

With `d = sum_l alpha_l*(mu_l**2 + var_l)` and `U = G @ m_comp.T`, the posterior covariance trace is:

```text
tr(P Bq) = d @ diag(G.T @ H @ G) - sum(U * (H @ U))
```

The diagonal floor and nonnegative trace clamp are retained. Tau statistics split weighted quadratic forms into within-donor centered terms and between-donor terms. Statistics are rebuilt after changes to the working weights, response, or posterior; caches are not reused across outer iterations.

## Validation

Install the optional test dependency and run from the repository root:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

Dense-reference tests cover unused donors, irregular cell ordering, strided and differently ordered genotype arrays, absent or redundant covariates, empty dimensions, sequential IBSS, and tau-objective behavior. Optional old-source comparisons cover full trajectories, credible sets, boundary freezing, and stopping rules.

To enable historical comparisons, set `SCELF_BASELINE_SRC` to an unmodified source snapshot that you maintain separately. For example, in PowerShell:

```powershell
$env:SCELF_BASELINE_SRC = (Resolve-Path ../scelf-reference/src).Path
python -m unittest discover -s tests -v
```

The sibling reference directory is user-supplied, not bundled or downloaded automatically. Without it, baseline-comparison tests skip; the independent dense-reference tests still run.

## Local performance measurements

With a reference snapshot configured, generate a local report:

```powershell
python benchmarks/benchmark_donor.py --baseline-src $env:SCELF_BASELINE_SRC --output benchmarks/timings.json
```

Generated reports are ignored by Git and are not shipped as evidence for a general speedup claim. Inspect numerical differences, iteration counts, BLAS thread settings, and first-call compilation costs when comparing timings. No fastmath, lower-precision arithmetic, or parallel component update is introduced.

## Known extreme-initialization limitation

A synthetic stress case with normal genotypes, `n=30, M=600, p=80, L=4, seed=43` produced an extremely large Stage 1 adaptive tau lower bound and unstable trajectories during development. Donor compression does not address the underlying initialization issue. Check parameter histories and finite values on new datasets rather than assuming that a completed fit is numerically well behaved. See [Fitting diagnostics](diagnostics.md).
