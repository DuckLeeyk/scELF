# Installation

scELF requires Python 3.10 or later. The distribution and import name are `scelf`; the public model class is `ScELF`.

## Install from the Git repository

In your chosen Python environment, clone the repository and install from its root directory. Replace `<repository-url>` with the repository URL:

```bash
git clone <repository-url> scELF
cd scELF
python -m pip install .
```

The package declares NumPy, pandas, SciPy, statsmodels, and Numba as runtime dependencies; pip installs them as needed.

For an editable installation when modifying the source:

```bash
python -m pip install -e .
```

Verify that the public interface is available:

```bash
python -c "from scelf import ScELF; print(ScELF)"
```

## Computation backend

`backend="auto"` selects the optional C++ extension if it loads successfully and otherwise uses Numba. The standard installation attempts to build the extension when possible; a native compiler is not required to use the Numba implementation.

```python
from scelf import backend_info

print(backend_info())
```

The native extension must match your Python version and platform and is built from source; compiled binaries are not included in the Git repository. Explicit `backend="cpp"` raises an error if the extension is unavailable, whereas `backend="numba"` explicitly selects Numba. The first Numba run can include compilation overhead. See the [C++ backend guide](performance-cpp.md) for native build requirements and controls.

## Run the bundled example

From the repository root:

```bash
python example/run_b_memory_lst1.py
```

See the [example guide](example.md) for inputs, iteration settings, and output files.

## Analysis dependencies

For the OneK1K command-line workflows, install the optional analysis and
comparator dependencies from the repository root:

```bash
python -m pip install -e ".[analysis,comparators]"
```

The `analysis` extra adds AnnData, scikit-learn, pandas-plink and xarray.
The `comparators` extra installs the pinned pySuSiE implementation used by the
SuSiE and SuSiE-RSS scripts. PLINK and the R environments for count normalization
and SAIGE-QTL are separate requirements. See [Analysis workflows](analysis-workflows.md)
for script descriptions and the analysis sequence.
