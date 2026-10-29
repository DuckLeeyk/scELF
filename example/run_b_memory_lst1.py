"""Run or benchmark scELF backends on the bundled B memory / LST1 data."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parent
# When running from a checkout, test its actual source rather than a stale install.
SOURCE_DIR = EXAMPLE_DIR.parent / "src"
if SOURCE_DIR.is_dir():
    sys.path.insert(0, str(SOURCE_DIR))

import scelf
from scelf import ScELF, backend_info


class TimedScELF(ScELF):
    """Record phase timings without changing the fitting algorithm."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.phase_seconds: dict[str, float] = {}

    def _stage1_warm_start(self, *args: Any, **kwargs: Any) -> None:
        start = time.perf_counter()
        try:
            return super()._stage1_warm_start(*args, **kwargs)
        finally:
            self.phase_seconds["stage1"] = time.perf_counter() - start

    def _stage2_variational_em(self, *args: Any, **kwargs: Any) -> None:
        start = time.perf_counter()
        try:
            return super()._stage2_variational_em(*args, **kwargs)
        finally:
            self.phase_seconds["stage2"] = time.perf_counter() - start


def read_genotype_ids(path: Path) -> list[str]:
    return pd.read_csv(path, header=None, sep=r"\s+", engine="python").iloc[:, 0].astype(str).tolist()


def read_donor_covariates(path: Path, donor_ids: list[str]) -> np.ndarray:
    covariates = pd.read_csv(path)
    if covariates.shape[1] < 2:
        raise ValueError("Donor covariates must contain an ID column and at least one covariate column.")

    if "genotype_ID" not in covariates.columns:
        covariates = covariates.rename(columns={covariates.columns[0]: "genotype_ID"})

    covariates["genotype_ID"] = covariates["genotype_ID"].astype(str)
    covariates = covariates.set_index("genotype_ID", drop=True)
    missing = [donor_id for donor_id in donor_ids if donor_id not in covariates.index]
    if missing:
        raise ValueError(f"Missing donor covariates for {len(missing)} donors, e.g. {missing[:5]}.")

    values = covariates.loc[donor_ids].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Donor covariates contain NaN or Inf.")
    return values


def read_cell_covariates(path: Path) -> np.ndarray:
    covariates = pd.read_csv(path)
    if covariates.shape[1] < 2:
        raise ValueError("Cell covariates must contain a cell ID column and at least one covariate column.")

    values = covariates.iloc[:, 1:].to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Cell covariates contain NaN or Inf.")
    return values


def read_design_matrix(path: Path, n_donors: int) -> np.ndarray:
    """Reconstruct the dense incidence matrix from zero-based donor indices."""
    donor_index = np.load(path, allow_pickle=False)
    if donor_index.ndim != 1 or donor_index.size == 0:
        raise ValueError("Donor indices must be a nonempty one-dimensional array.")
    if not np.issubdtype(donor_index.dtype, np.integer):
        raise ValueError("Donor indices must have an integer dtype.")
    if n_donors < 1 or np.any(donor_index < 0) or np.any(donor_index >= n_donors):
        raise ValueError("Donor indices must be in the range [0, n_donors).")
    design = np.zeros((donor_index.size, n_donors), dtype=np.uint8)
    design[np.arange(donor_index.size), donor_index] = 1
    return design


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict()
    raise TypeError(f"Cannot serialize {type(value)!r} to JSON.")


def load_example_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    donor_ids = read_genotype_ids(EXAMPLE_DIR / "genotype_IDs_B_memory.txt")
    Xd = read_donor_covariates(EXAMPLE_DIR / "individual_covariates_dummy.csv", donor_ids)
    Xc = read_cell_covariates(EXAMPLE_DIR / "cell_covariates_B memory.csv")
    Z = read_design_matrix(EXAMPLE_DIR / "cell_donor_index_B_memory.npy", len(donor_ids))

    y = np.asarray(np.loadtxt(EXAMPLE_DIR / "GReX_B_memory_LST1.txt", dtype=float)).reshape(-1)
    if y.shape[0] != Z.shape[0] or Xc.shape[0] != Z.shape[0]:
        raise ValueError("Counts, cell covariates, and design matrix must have the same number of cells.")
    if not np.isfinite(y).all():
        raise ValueError("Expression counts contain NaN or Inf.")

    genotype = pd.read_csv(EXAMPLE_DIR / "genotype_B_memory_LST1.csv", index_col=0)
    variant_ids = genotype.columns.astype(str).to_numpy()
    G = genotype.to_numpy(dtype=float)
    if G.shape[0] != len(donor_ids):
        raise ValueError("Genotype rows and donor IDs are not aligned.")
    if not np.isfinite(G).all():
        raise ValueError("Genotypes contain NaN or Inf.")

    genotype_sd = G.std(axis=0, ddof=0)
    if np.any(genotype_sd == 0):
        raise ValueError("Genotypes contain one or more zero-variance variants.")
    G = np.asfortranarray((G - G.mean(axis=0)) / genotype_sd)

    return y, Xc, Xd, Z, G, variant_ids


def fit_once(data: tuple, backend: str, args: argparse.Namespace, iteration_mode: str) -> tuple:
    y, Xc, Xd, Z, G, _ = data
    start = time.perf_counter()
    model = TimedScELF(y=y, Xc=Xc, Xd=Xd, Z=Z, G=G, L=10, phi=None, seed=args.seed, backend=backend)
    initialization_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model.fit(
        use_covariates=True,
        max_outer_iter=args.max_outer_iter,
        tol_F=-np.inf if iteration_mode == "fixed" else 1e-5,
        verbose=not args.quiet and not args.compare_backends,
        stage1_options={"estimate_phi": True, "phi_min": 1e-4, "phi_max": 1e4, "phi_init": 100},
        stage2_options={
            "tau": {"method": "bounded"},
            "phi": {"update": True, "phi_min": 1e-4, "phi_max": 1e4},
            "e_step": {"ibss_sweeps": 1, "update_prior_var": True},
        },
    )
    elapsed = time.perf_counter() - start
    return model, {
        "requested_backend": backend, "actual_backend": model.backend,
        "initialization_seconds": initialization_seconds, "fit_seconds": elapsed,
        "stage1_seconds": model.phase_seconds["stage1"],
        "stage2_seconds": model.phase_seconds["stage2"],
        "result_processing_seconds": max(0., elapsed - sum(model.phase_seconds.values())),
        "outer_iterations": len(model.history["F"]),
        "final_elbo": model.history["F"][-1], "tau": model.tau, "phi": model.phi,
        "finite_elbo_history": bool(np.isfinite(model.history["F"]).all()),
        "finite_posterior": all(bool(np.isfinite(value).all()) for value in model.susie_q.values()),
    }


def write_model_results(model: ScELF, variant_ids: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variant_id": variant_ids, "pip": np.asarray(model.pip, dtype=float)}).to_csv(
        output_dir / "B_memory_LST1_pip.csv", index=False
    )
    with (output_dir / "B_memory_LST1_credible_sets.json").open("w", encoding="utf-8") as handle:
        json.dump(model.sets, handle, indent=2, default=to_jsonable)


def json_safe(value: Any) -> Any:
    """Use null for nonfinite diagnostics, keeping the performance report valid JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return json_safe(value.tolist() if isinstance(value, np.ndarray) else value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def max_abs_difference(a: Any, b: Any) -> float | None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(np.max(np.abs(a - b))) if a.size else 0.


def compare_results(numba: ScELF, cpp: ScELF) -> dict:
    # The accessor updates model.pip, so preserve the exported, pruned PIP.
    def unpruned(model: ScELF) -> np.ndarray:
        saved = model.pip.copy()
        try:
            return model.get_variant_pips(prune_by_cs=False).copy()
        finally:
            model.pip = saved
    n_pip, c_pip = unpruned(numba), unpruned(cpp)
    n_cs = numba.sets.get("cs") or {}
    c_cs = cpp.sets.get("cs") or {}
    same_cs = n_cs.keys() == c_cs.keys() and all(
        np.array_equal(np.sort(n_cs[key]), np.sort(c_cs[key])) for key in n_cs
    )
    return {
        "max_abs_pruned_pip_difference": max_abs_difference(numba.pip, cpp.pip),
        "max_abs_unpruned_pip_difference": max_abs_difference(n_pip, c_pip),
        "unpruned_pips_allclose": bool(np.allclose(n_pip, c_pip, rtol=1e-5, atol=1e-7)),
        "credible_sets_equal": same_cs,
        "same_outer_iterations": len(numba.history["F"]) == len(cpp.history["F"]),
        "max_abs_elbo_history_difference": max_abs_difference(numba.history["F"], cpp.history["F"]),
        "max_abs_tau_history_difference": max_abs_difference(numba.history["tau"], cpp.history["tau"]),
        "max_abs_phi_history_difference": max_abs_difference(numba.history["phi"], cpp.history["phi"]),
        "posterior_max_abs_differences": {
            key: max_abs_difference(numba.susie_q[key], cpp.susie_q[key]) for key in numba.susie_q
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-outer-iter",
        type=int,
        default=1,
        help="Number of Stage 2 variational-EM iterations (default: 1 for a smoke test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "output",
        help="Directory for PIP and credible-set output files.",
    )
    parser.add_argument("--backend", choices=("auto", "numba", "cpp"), default="auto",
                        help="Backend for a single run (default: auto).")
    parser.add_argument("--compare-backends", action="store_true",
                        help="Benchmark explicit numba and cpp backends on the same data and seed.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Measured repetitions (default: 3 for comparison, 1 otherwise).")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Untimed full fits per backend (default: 1 for comparison, 0 otherwise).")
    parser.add_argument("--blas-threads", type=int, default=None,
                        help="Limit BLAS threads (comparison defaults to 1; single runs preserve existing settings).")
    parser.add_argument("--iteration-mode", choices=("fixed", "converged"), default=None,
                        help="Comparison defaults to fixed iterations; single runs retain convergence stopping.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Suppress model progress messages in single runs.")
    args = parser.parse_args(argv)
    repeats = args.repeats if args.repeats is not None else (3 if args.compare_backends else 1)
    warmup = args.warmup if args.warmup is not None else (1 if args.compare_backends else 0)
    threads = args.blas_threads if args.blas_threads is not None else (1 if args.compare_backends else None)
    mode = args.iteration_mode or ("fixed" if args.compare_backends else "converged")
    if args.max_outer_iter < 1 or repeats < 1 or warmup < 0 or (threads is not None and threads < 1):
        parser.error("iterations, repeats and thread count must be positive; warmup must be nonnegative")
    if args.compare_backends and args.backend != "auto":
        parser.error("--compare-backends already selects both backends; omit --backend")
    info = backend_info()
    if (args.compare_backends or args.backend == "cpp") and not info["cpp_available"]:
        parser.error(f"C++ backend is unavailable: {info['cpp_error']}. Rebuild it with this Python interpreter.")
    thread_context = nullcontext()
    threadpool_info = lambda: []
    if threads is not None:
        try:
            from threadpoolctl import threadpool_limits, threadpool_info
        except ImportError:
            parser.error("--blas-threads/comparison requires threadpoolctl: python -m pip install threadpoolctl")
        thread_context = threadpool_limits(limits=threads)

    print(f"Python: {sys.executable}")
    print(f"scelf source: {scelf.__file__}")
    print(f"Backend availability: {info}")
    start = time.perf_counter()
    data = load_example_data()
    load_seconds = time.perf_counter() - start
    y, Xc, Xd, Z, G, variant_ids = data

    print(f"Loaded B memory / LST1 data: {y.shape[0]} cells, {G.shape[0]} donors, {G.shape[1]} variants.")

    backends = ["numba", "cpp"] if args.compare_backends else [args.backend]
    print(f"Mode={mode}; warmup={warmup}; repetitions={repeats}; BLAS threads={threads or 'existing settings'}.")
    print("Fit timings exclude data loading, model construction and output writing.")
    measurements = {backend: [] for backend in backends}
    last_models = {}
    report = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "scelf_source": scelf.__file__, "numpy": np.__version__, "backend_info": info,
        "data_loading_seconds": load_seconds,
        "data": {"cells": y.size, "donors": G.shape[0], "variants": G.shape[1],
                 "cell_covariates": Xc.shape[1], "donor_covariates": Xd.shape[1]},
        "settings": {"iteration_mode": mode, "max_outer_iter": args.max_outer_iter,
                     "seed": args.seed, "repeats": repeats, "warmup": warmup, "blas_threads": threads},
    }
    with thread_context:
        report["threadpools"] = threadpool_info()
        for round_index in range(warmup + repeats):
            # Alternate backend order across rounds to reduce order/thermal bias.
            order = backends if round_index % 2 == 0 else backends[::-1]
            is_warmup = round_index < warmup
            for backend in order:
                label = "warmup" if is_warmup else "measured"
                print(f"[{backend}] {label} run {round_index + 1}/{warmup + repeats} ...", flush=True)
                model, timing = fit_once(data, backend, args, mode)
                print(f"  actual={model.backend}; fit={timing['fit_seconds']:.3f}s; "
                      f"Stage 1={timing['stage1_seconds']:.3f}s; Stage 2={timing['stage2_seconds']:.3f}s; "
                      f"outer iterations={timing['outer_iterations']}", flush=True)
                if not is_warmup:
                    measurements[backend].append(timing)
                    last_models[backend] = model
        report["backends"] = {}
        for backend in backends:
            times = measurements[backend]
            median = statistics.median(item["fit_seconds"] for item in times)
            report["backends"][backend] = {"actual_backend": last_models[backend].backend,
                                          "median_fit_seconds": median, "runs": times}
            destination = args.output_dir / backend if args.compare_backends else args.output_dir
            write_model_results(last_models[backend], variant_ids, destination)
            print(f"[{backend}] median fit: {median:.3f}s; results: {destination}")
        if args.compare_backends:
            ratio = report["backends"]["numba"]["median_fit_seconds"] / report["backends"]["cpp"]["median_fit_seconds"]
            comparison = compare_results(last_models["numba"], last_models["cpp"])
            iteration_counts = [item["outer_iterations"] for items in measurements.values() for item in items]
            report["comparison"] = {"numba_over_cpp_median_fit_ratio": ratio,
                                    "same_iteration_count_all_runs": len(set(iteration_counts)) == 1,
                                    **comparison}
            pd.DataFrame({"variant_id": variant_ids, "pip_numba": last_models["numba"].pip,
                          "pip_cpp": last_models["cpp"].pip,
                          "absolute_difference": np.abs(last_models["numba"].pip - last_models["cpp"].pip)}).to_csv(
                args.output_dir / "B_memory_LST1_backend_comparison.csv", index=False
            )
            print(f"Numba / C++ median fit ratio: {ratio:.2f}x (>1 means C++ is faster).")
            print(f"Result comparison: {comparison}")
            if not report["comparison"]["same_iteration_count_all_runs"]:
                print("Iteration counts differ: interpret the timing ratio together with convergence and result differences.")
            if not comparison["unpruned_pips_allclose"] or not comparison["credible_sets_equal"]:
                print("Numerical results differ; inspect the PIP comparison and posterior/history diagnostics.")
            if any(not item["finite_elbo_history"] or not item["finite_posterior"] for items in measurements.values() for item in items):
                print("A fit produced nonfinite diagnostics; check the report before interpreting its timing.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    performance_path = args.output_dir / "B_memory_LST1_performance.json"
    with performance_path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, allow_nan=False)
    print(f"Performance report written to {performance_path}.")


if __name__ == "__main__":
    main()
