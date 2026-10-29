# PIP-based colocalization

`coloc_pip` is an additional scELF package feature for comparing two aligned
single-signal PIP vectors. It is independent of fitting a `ScELF` model.

```python
import numpy as np
from scelf import coloc_pip

# Same variants in the same order on both sides.
eqtl = np.array([0.70, 0.20, 0.10])
gwas = np.array([0.60, 0.35, 0.05])
result = coloc_pip(eqtl, gwas, p1=1e-4, p2=1e-4, p12=1e-7)
print(result.h4)
```

For `Se = sum(eqtl)`, `Sg = sum(gwas)`, and `Seg = sum(eqtl * gwas)`, the
unnormalized weights are:

```text
W3 = p1 * p2 * (Se * Sg - Seg)
W4 = p12 * Seg
H4 = W4 / (W3 + W4)
H3 = 1 - H4
```

This is conditional support for shared-variant H4 versus distinct-variant H3.
It does not estimate H0, H1 or H2, and is not a five-hypothesis coloc posterior.
The calculation retains the original `model/coloc_pip.py` log-space algorithm
and prior defaults; different Gentropy releases may use different defaults.

## API

```python
coloc_pip(pip1, pip2, p1=1e-4, p2=1e-4, p12=1e-7,
          pseudocount=1e-16, validate=True) -> ColocPIPResult
```

Inputs must be finite, nonempty, equally sized 1-D arrays, with each PIP in
`[0,1]` and positive total support on both sides. Prior weights must be finite
and in `(0,1]`; the numerical floor must be in `(0,1)`. Validation raises
`ValueError`. Each PIP is lower-bounded by the pseudocount before computation,
so zero padding introduces only that numerical floor. Vectors are not normalized
by the function; restricting marginal PIPs to a CS is supported. Keep validation
enabled unless the caller guarantees these contracts.

The immutable result has fields `h3`, `h4`, `score` (equal to `h4`), `log_pp3`,
and `log_pp4` (unnormalized log weights). Result fields are Python floats.

The caller must align genome builds, variants and credible sets. The package
function does not read files, infer alleles, construct CSs, fill residual mass,
or determine overlap. For manuscript-style union/zero alignment, form the full
union of both CSs and pad absent PIPs with zero before calling. Apply this to
CS pairs with at least one shared variant. Use single-signal probabilities when
available; marginal PIPs across multiple signals require careful interpretation.

The [OneK1K analysis workflows](analysis-workflows.md) introduce the command-line
scripts for preparation, four-method fine-mapping, BCX2/FinnGen colocalization,
TWAS and locus exports. The checkout's `analysis/onek1k/README.md` provides full
commands, including the explicit choice between historical and manuscript PIP
policies.
