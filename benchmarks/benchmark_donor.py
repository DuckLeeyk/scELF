"""Reproducible warmed, single-BLAS-thread comparison against the old source.

python benchmarks/benchmark_donor.py --baseline-src PATH --output timings.json
The baseline path contains the original scelf package directory.
"""
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
import numba
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_donor_compression import ScELF, fixture, load_baseline


def measure(action, reset=lambda: None, repeats=3):
    reset()
    action()  # JIT/BLAS warmup excluded from timings.
    values = []
    for _ in range(repeats):
        reset()
        start = time.perf_counter()
        action()
        values.append(time.perf_counter() - start)
    return {'median_s': statistics.median(values), 'samples_s': values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-src', required=True)
    parser.add_argument('--output', type=Path, default=Path('timings.json'))
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    reference = load_baseline(args.baseline_src)
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'numba': numba.__version__, 'scipy': scipy.__version__,
              'blas_threads': 1, 'jit': 'warmed; first call excluded', 'cases': []}
    with threadpool_limits(limits=1):
        report['threadpools'] = threadpool_info()
        for n, cells, p, L in [(100, 10000, 1000, 6), (200, 40000, 3000, 10)]:
            case = {'n': n, 'cells': cells, 'p': p, 'L': L, 'covariates': 3, 'phases': {}}
            # Keep this historical benchmark pinned to the Numba baseline.
            for label, cls in [('baseline', reference), ('optimized', partial(ScELF, backend='numba'))]:
                m, X = fixture(cls, n=n, cells=cells, p=p, L=L)
                q0 = copy.deepcopy(m.susie_q)
                sum0 = m.m_sum.copy()

                def reset():
                    m.susie_q = copy.deepcopy(q0)
                    m.m_sum = sum0.copy()
                    m.tau = .3
                    m.tau_frozen_at_lower_bound = False
                    m.rng = np.random.default_rng(42)

                solver = m._make_exact_solver()
                cache = m._make_projection_cache(solver, X)

                def trace():
                    # Include diagonal computation, as with a fresh ELBO cache.
                    cache.pop('diag_GPG', None)
                    return m._trace_P_Bq_exact(solver, X, cache)

                actions = {
                    'ibss_sweep': lambda: m._ibss_full_cycle_collapsed(X),
                    'trace': trace,
                    'tau_bounded_search': lambda: m._update_tau_direct(X, {'tau': {'tau_max': 100.0}}),
                    'credible_sets': lambda: m.get_credible_sets(),
                }
                if label == 'optimized':
                    st = m._prepare_tau_statistics(X)
                    actions['tau_objective_prepared'] = lambda: m._collapsed_variational_reml_Q_tau(X, .5, st)
                    actions['tau_prepare'] = lambda: m._prepare_tau_statistics(X)
                else:
                    actions['tau_objective_prepared'] = lambda: m._collapsed_variational_reml_Q_tau(X, .5)
                for phase, action in actions.items():
                    timing = measure(action, reset, args.repeats)
                    case['phases'].setdefault(phase, {})[label] = timing
                    print(f'{n=}, {cells=}, {p=}: {label} {phase} {timing["median_s"]:.6f}s', flush=True)

                def fresh_fit():
                    fresh, _ = fixture(cls, n=n, cells=cells, p=p, L=L)
                    fresh.fit(max_outer_iter=5, tol_F=-np.inf, verbose=False,
                              stage2_options={'tau': {'tau_max': 100.0}})
                    return fresh
                timing = measure(fresh_fit, repeats=args.repeats)
                case['phases'].setdefault('full_fit_5_outer_iterations', {})[label] = timing
                fitted = fresh_fit()
                timing['outer_iterations'] = len(fitted.history['F'])
                timing['final_elbo'] = fitted.history['F'][-1]
                timing['tau'] = fitted.tau
                print(f'{label} full_fit {timing["median_s"]:.6f}s', flush=True)
            for phase, timings in case['phases'].items():
                if 'baseline' in timings and 'optimized' in timings:
                    timings['speedup'] = timings['baseline']['median_s'] / timings['optimized']['median_s']
            report['cases'].append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
