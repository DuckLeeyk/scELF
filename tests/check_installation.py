"""Smoke-check the installed distribution, without importing checkout source.

Run with `python -I tests/check_installation.py --expected-backend numba`.
"""

import argparse
from importlib.metadata import metadata
from pathlib import Path

import numpy as np
import scelf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-backend", choices=("numba", "cpp"), required=True)
    args = parser.parse_args()

    checkout_source = Path(__file__).resolve().parents[1] / "src"
    package_path = Path(scelf.__file__).resolve()
    if package_path.is_relative_to(checkout_source):
        raise AssertionError("Expected a non-editable installation, not checkout source.")
    info = scelf.backend_info()
    assert info["auto_backend"] == args.expected_backend, info
    assert info["cpp_available"] == (args.expected_backend == "cpp"), info
    assert metadata("scelf")["License-Expression"] == "GPL-3.0-only"

    rng = np.random.default_rng(23)
    donor = np.repeat(np.arange(10), 4)
    G = rng.binomial(2, 0.4, (10, 8)).astype(float)
    G = (G - G.mean(axis=0)) / G.std(axis=0)
    y = rng.poisson(np.exp(0.2 + 0.2 * G[donor, 0]))
    model = scelf.ScELF(
        y, np.ones((40, 1)), None, np.eye(10)[donor], G,
        L=2, phi=5, seed=42, backend="auto",
    )
    model.fit(max_outer_iter=2, verbose=False)
    assert model.backend == args.expected_backend
    assert model.pip.shape == (8,) and np.isfinite(model.pip).all()
    assert len(model.history["F"]) == 2 and np.isfinite(model.history["F"]).all()
    assert 0 <= scelf.coloc_pip([0.7, 0.3], [0.8, 0.2]).h4 <= 1
    print(f"Installed package smoke check passed: backend={model.backend}.")


if __name__ == "__main__":
    main()
