# Optional C++ backend

The native backend extends the Numba/donor-compression implementation with C++17 kernels for sequential Stage 2 IBSS, genotype projection diagonals, posterior covariance traces, repeated tau-objective calculations, and Stage 1 prior-variance likelihood evaluation. It uses pybind11 and float64 arithmetic.

NumPy/SciPy still handle general linear algebra and optimization, and Numba still handles cell aggregation and the Woodbury solver. No OpenMP worker pool is added. The native IBSS uses SciPy's BLAS and releases the GIL during computation. Use independent model instances for independent workers; do not fit the same instance concurrently.

## Selection and availability

`ScELF(..., backend="auto")` uses C++ if the extension loads successfully and otherwise uses Numba. Explicit C++ selection raises an `ImportError` when unavailable.

```python
from scelf import ScELF, backend_info

print(backend_info())
model = ScELF(y, Xc, Xd, Z, G, L=10, backend="cpp")
model.fit()
print(model.backend)
```

The standalone `scelf.susie.SuSiE` class accepts the same option, with a default of `"numba"`. ScELF passes its selected backend to the Stage 1 initializer.

Native boundaries check float64 layout, shape, writable outputs, overlapping posterior buffers, and BLAS dimension ranges. The extension obtains BLAS through SciPy's exported C API, without a hard-coded DLL path. Incompatible SciPy capsule signatures result in an import error and automatic Numba fallback in auto mode.

## Build from source

Run commands from the repository root. Standard Windows builds require Microsoft C++ Build Tools and the Windows SDK; Linux/macOS builds require a C++17 compiler. pybind11 is a build dependency, not a runtime dependency. Default installation attempts an optional extension and permits Numba-only use if compilation is unavailable.

To require a native build in PowerShell:

```powershell
$env:SCELF_BUILD_CPP = '1'
python -m pip install -e .
```

On Linux/macOS:

```bash
SCELF_BUILD_CPP=1 python -m pip install -e .
```

Set `SCELF_BUILD_CPP=0` instead to skip the native build. Build the extension with the Python interpreter and platform that will run it; compiled binaries are not distributed in the Git repository. Inspect `backend_info()` after installation.

### Optional Zig compiler on Windows

If you already have a compatible Zig installation, set `SCELF_ZIG` to your own compiler executable:

```powershell
$env:SCELF_ZIG = (Get-Command zig -ErrorAction Stop).Source
$env:SCELF_BUILD_CPP = '1'
python -m pip install -e .
```

This alternative build path supports Windows only. It does not install a compiler or depend on a project-specific tools directory. Do not rename compiled binaries to bypass Python ABI tags.

## Tests and local measurements

Install the test dependencies and run from the repository root:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

Tests compare explicit backend selection, missing-extension fallback, dense references, sequential posterior updates, standalone SuSiE, likelihood/prior optimization, and native boundary validation. Native tests skip when the extension is absent; CI has separate jobs that require a successful C++ build, so missing extensions cannot silently pass those jobs. Historical baseline comparisons require an independently supplied reference snapshot and otherwise skip.

Generate your own local timing report:

```bash
python benchmarks/benchmark_cpp.py --output benchmarks/cpp-timings.json
```

Generated JSON reports are ignored by Git because they contain environment-specific timings and paths. Compare outputs and iteration counts alongside timings; fixed-iteration synthetic measurements do not establish real-data time-to-convergence.

No fastmath or reduced precision is enabled. Reduction order, BLAS libraries, and native math libraries can introduce small floating-point differences, so numerical tests use tolerances rather than bitwise equality. See [Donor compression](performance-numba-donor.md) for the solver structure and known initialization limitation.
