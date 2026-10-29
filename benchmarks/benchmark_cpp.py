"""Compare native and optimized Numba backends, with identical work and BLAS limits."""
import argparse
import copy
from functools import partial
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
import scipy
import numba
from threadpoolctl import threadpool_limits, threadpool_info
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_donor_compression import fixture, ScELF
from scelf import backend_info


def measure(action, reset, repeats):
    reset(); action()
    samples = []
    for _ in range(repeats):
        reset()
        start = time.perf_counter()
        action()
        samples.append(time.perf_counter() - start)
    return {'median_s': statistics.median(samples), 'samples_s': samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/cpp-timings.json'))
    parser.add_argument('--blas-threads', type=int, default=1)
    args = parser.parse_args()
    if not backend_info()['cpp_available']:
        raise RuntimeError('Build the native extension before benchmarking.')
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'numba': numba.__version__,
              'backend_info': backend_info(), 'blas_threads': args.blas_threads,
              'jit': 'warmed; excluded from timing', 'cases': []}
    with threadpool_limits(limits=args.blas_threads):
        report['threadpools'] = threadpool_info()
        for n, M, p, L in [(100, 10000, 1000, 6), (200, 40000, 3000, 10)]:
            case = {'n': n, 'M': M, 'p': p, 'L': L, 'covariates': 3, 'phases': {}, 'accuracy': {}}
            fits = {}
            for backend in ['numba', 'cpp']:
                constructor = partial(ScELF, backend=backend)
                model, X = fixture(constructor, n=n, cells=M, p=p, L=L)
                original_q = copy.deepcopy(model.susie_q)
                original_total = model.m_sum.copy()
                original_D = model.D.copy()
                original_response = model.y_work.copy()
                def reset():
                    model.susie_q = copy.deepcopy(original_q)
                    model.m_sum = original_total.copy()
                    model.tau = .3
                    model.tau_frozen_at_lower_bound = False
                    model.tau_freeze_value = None
                    model.tau_lower_bound = 1e-12
                    model.phi = 8.
                    model.history = {key: [] for key in model.history}
                    model.D = original_D.copy()
                    model.y_work = original_response.copy()
                solver = model._make_exact_solver()
                cache = model._make_projection_cache(solver, X)
                stats = model._prepare_tau_statistics(X)
                def trace():
                    cache.pop('diag_GPG', None)
                    return model._trace_P_Bq_exact(solver, X, cache)
                actions = {
                    'ibss_sweep': lambda: model._ibss_full_cycle_collapsed(X),
                    'trace': trace,
                    'tau_objective_prepared': lambda: model._collapsed_variational_reml_Q_tau(X, .5, stats),
                    'tau_bounded_search': lambda: model._update_tau_direct(X, {'tau': {'tau_max': 100.0}}),
                    'stage1_warm_start': lambda: model._stage1_warm_start({}, verbose=False),
                }
                for phase, action in actions.items():
                    timing = measure(action, reset, args.repeats)
                    case['phases'].setdefault(phase, {})[backend] = timing
                    print(f'{n=} {p=} {backend} {phase}: {timing["median_s"]*1000:.3f} ms', flush=True)
                def full_fit():
                    fresh, _ = fixture(constructor, n=n, cells=M, p=p, L=L)
                    fresh.fit(max_outer_iter=5, tol_F=-np.inf, verbose=False,
                              stage2_options={'tau': {'tau_max': 100.0}})
                    return fresh
                timing = measure(full_fit, lambda: None, args.repeats)
                fitted = full_fit()
                fits[backend] = fitted
                timing['outer_iterations'] = len(fitted.history['F'])
                case['phases'].setdefault('full_fit_5_outer_iterations', {})[backend] = timing
                print(f'{backend} full fit: {timing["median_s"]*1000:.3f} ms', flush=True)
            for phase in case['phases'].values():
                phase['speedup'] = phase['numba']['median_s'] / phase['cpp']['median_s']
            a, b = fits['numba'], fits['cpp']
            case['accuracy'] = {
                'max_abs_unpruned_pip_diff': float(np.max(np.abs(a.get_variant_pips(False) - b.get_variant_pips(False)))),
                'max_abs_elbo_diff': float(np.max(np.abs(np.asarray(a.history['F']) - b.history['F']))),
                'max_abs_tau_diff': float(np.max(np.abs(np.asarray(a.history['tau']) - b.history['tau']))),
                'posterior_max_abs_differences': {key: float(np.max(np.abs(a.susie_q[key] - b.susie_q[key]))) for key in a.susie_q},
            }
            report['cases'].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
