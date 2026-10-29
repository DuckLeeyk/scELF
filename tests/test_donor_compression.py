"""Numerical regression tests: python -m unittest discover -s tests -v.

Set SCELF_BASELINE_SRC to an unmodified src directory for full-fit comparisons.
The dense reference tests do not require the old implementation.
"""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scelf import ScELF
from scelf.solver import ExactVSolver
from scelf.numerics import _compute_s_vec_numba, _Gt_times_vec_numba


def load_baseline(path):
    root = Path(path)
    # Accept a source root or the package itself, including historical snapshots.
    packages = ([root] if (root / '__init__.py').is_file() else
                [p for p in root.iterdir() if p.is_dir()
                 and (p / '__init__.py').is_file() and (p / 'model.py').is_file()])
    if len(packages) != 1:
        raise ValueError(f'Expected one model package under {root}, found {len(packages)}')
    package = packages[0]
    spec = importlib.util.spec_from_file_location(
        'scelf_reference', package / '__init__.py', submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, module.__all__[0])


def fixture(cls=ScELF, n=7, cells=43, p=13, L=3, cov=True, seed=42, genotype_scale=1.0, effect=.2):
    rng = np.random.default_rng(seed)
    # Noncontiguous cells, unbalanced donors, and an unused donor.
    ids = rng.integers(0, n - 1, size=cells)
    Z = np.eye(n)[ids]
    G = genotype_scale * rng.normal(size=(n, p))
    Xc = np.column_stack((np.ones(cells), rng.normal(size=cells))) if cov else None
    Xd = rng.normal(size=(n, 1)) if cov else None
    y = rng.poisson(np.exp(0.2 + effect * G[ids, 0]))
    m = cls(y, Xc, Xd, Z, G, L, phi=8.0, tau=0.3, tau_fixed=False, seed=seed)
    m.D = rng.uniform(0.2, 2.0, cells)
    m.y_work = rng.normal(size=cells)
    q = m.susie_q
    q['pip_comp'] = rng.dirichlet(np.ones(p), size=L)
    q['post_mean'] = rng.normal(size=(L, p))
    q['post_var'] = rng.uniform(0.1, 1.0, size=(L, p))
    q['m_comp'] = q['pip_comp'] * q['post_mean']
    q['prior_var'] = rng.uniform(0.2, 1.0, L)
    m.m_sum = q['m_comp'].sum(axis=0)
    X = m._build_design_X() if cov else np.zeros((cells, 0))
    return m, X


def dense_projection(m, X, tau):
    W = np.maximum(m.D, 1e-12)
    Sigma = np.diag(1 / W) + tau * (m.Z @ m.Z.T)
    V = np.linalg.inv(Sigma)
    gram = X.T @ V @ X + 1e-10 * np.eye(X.shape[1])
    inv = np.linalg.inv(gram)
    P = V - V @ X @ inv @ X.T @ V
    return P, Sigma, gram


def dense_trace(m, P):
    q = m.susie_q
    d = np.sum(q['pip_comp'] * (q['post_mean'] ** 2 + q['post_var']), axis=0)
    Gcell = m.Z @ m.G
    diagonal = np.maximum(np.diag(Gcell.T @ P @ Gcell), 1e-12)
    U = Gcell @ q['m_comp'].T
    return max(float(d @ diagonal - np.sum(U * (P @ U))), 0.0)


def dense_ibss(m, X, update_prior=True):
    P, _, _ = dense_projection(m, X, m.tau)
    Gcell = m.Z @ m.G
    diagonal = np.maximum(np.diag(Gcell.T @ P @ Gcell), 1e-12)
    for l in range(m.L):
        minus = m.m_sum - m.susie_q['m_comp'][l]
        a = Gcell.T @ P @ (m.y_work - Gcell @ minus)
        sigma2 = max(float(m.susie_q['prior_var'][l]), 1e-12)
        var = np.maximum(1 / np.maximum(diagonal + 1 / sigma2, 1e-20), 1e-18)
        mu = var * a
        logits = -np.log(m.p) + 0.5 * (np.log(var) - np.log(sigma2)) + 0.5 * mu**2 / np.maximum(var, 1e-18)
        weights = np.exp(logits - logits.max())
        pip = weights / weights.sum()
        pip = np.clip(pip, 0, 1)
        pip /= max(float(pip.sum()), 1e-20)
        for key, value in [('pip_comp', pip), ('post_mean', mu), ('post_var', var), ('m_comp', pip * mu)]:
            m.susie_q[key][l] = value
        m.m_sum = minus + pip * mu
        if update_prior:
            m.susie_q['prior_var'][l] = max(float(np.sum(pip * (mu**2 + var))), 1e-12)


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_solver_vector_matrix_and_strides(self):
        m, X = fixture()
        _, Sigma, _ = dense_projection(m, X, m.tau)
        solver = m._make_exact_solver()
        assert_allclose(solver.solve(m.y_work[::-1]), np.linalg.solve(Sigma, m.y_work[::-1]), rtol=1e-11, atol=1e-11)
        assert_allclose(solver.solve_matrix(X), np.linalg.solve(Sigma, X), rtol=1e-11, atol=1e-11)
        assert_allclose(solver.logdet_sigma(), np.linalg.slogdet(Sigma)[1], atol=1e-11)

    def test_genotype_kernels_layouts(self):
        m, _ = fixture()
        for G in [np.array(m.G, order='C'), np.array(m.G, order='F'), m.G[:, ::2]]:
            assert_allclose(_Gt_times_vec_numba(G, m.C + 1), G.T @ (m.C + 1), atol=1e-12)
            assert_allclose(_compute_s_vec_numba(G, m.C + 1), (G**2).T @ (m.C + 1), atol=1e-12)
        self.assertTrue(m.G.flags.f_contiguous)

    def test_donor_operator_diag_and_traces(self):
        for cov in [False, True]:
            m, X = fixture(cov=cov)
            for tau in [1e-8, 0.3, 100.0]:
                m.tau = tau
                solver = m._make_exact_solver()
                cache = m._make_projection_cache(solver, X)
                P, _, _ = dense_projection(m, X, tau)
                H = m.Z.T @ P @ m.Z
                v = np.arange(m.n, dtype=float)
                assert_allclose(m._apply_donor_P(v, solver, cache), H @ v, rtol=1e-9, atol=1e-9)
                assert_allclose(m._apply_donor_P(np.column_stack((v, v + 1)), solver, cache), H @ np.column_stack((v, v + 1)), rtol=1e-9, atol=1e-9)
                Gcell = m.Z @ m.G
                assert_allclose(m._compute_diag_GPG(solver, X, cache), np.maximum(np.diag(Gcell.T @ P @ Gcell), 1e-12), rtol=1e-9, atol=1e-9)
                assert_allclose(m._trace_P_Bq_exact(solver, X, cache), dense_trace(m, P), rtol=1e-9, atol=1e-9)
                assert_allclose(m._trace_PK_exact(solver, X, cache), np.trace(H), rtol=1e-9, atol=1e-9)

    def test_tau_objective_and_no_state_mutation(self):
        for cov in [False, True]:
            m, X = fixture(cov=cov)
            st = m._prepare_tau_statistics(X)
            original_tau = m.tau
            for tau in [1e-8, 0.3, 100.0]:
                P, Sigma, gram = dense_projection(m, X, tau)
                z = m.y_work - m.Z @ m.G @ m.m_sum
                expected = -0.5 * (np.linalg.slogdet(Sigma)[1] + np.linalg.slogdet(gram)[1] + z @ P @ z + dense_trace(m, P))
                assert_allclose(m._collapsed_variational_reml_Q_tau(X, tau, st), expected, rtol=1e-9, atol=1e-9)
                self.assertEqual(m.tau, original_tau)

    def test_ibss_against_dense_sequential_reference(self):
        for cov in [False, True]:
            for update_prior in [False, True]:
                m, X = fixture(cov=cov)
                ref = copy.deepcopy(m)
                for _ in range(3):
                    m._ibss_full_cycle_collapsed(X, update_prior)
                    dense_ibss(ref, X, update_prior)
                    for key in m.susie_q:
                        assert_allclose(m.susie_q[key], ref.susie_q[key], rtol=2e-9, atol=2e-10)
                    assert_allclose(m.m_sum, ref.m_sum, rtol=2e-9, atol=2e-10)

    def test_rank_deficient_covariates(self):
        m, _ = fixture()
        X = np.column_stack((np.ones(m.M), np.ones(m.M)))
        solver = m._make_exact_solver()
        cache = m._make_projection_cache(solver, X)
        cell = m.Z @ np.arange(m.n, dtype=float)
        assert_allclose(m._apply_donor_P(np.arange(m.n, dtype=float), solver, cache), m.Z.T @ m._apply_P(cell, solver, X, cache), rtol=2e-5, atol=2e-5)
        self.assertTrue(np.isfinite(m._collapsed_variational_reml_Q_tau(X, 1e6)))

    def test_empty_genotypes_and_components(self):
        for p, L in [(0, 3), (4, 0)]:
            m = ScELF(np.ones(6), None, None, np.eye(3)[[0, 1, 2, 0, 1, 2]], np.zeros((3, p)), L, phi=8, tau=.3)
            X = np.zeros((6, 0))
            m._ibss_full_cycle_collapsed(X)
            solver = m._make_exact_solver()
            cache = m._make_projection_cache(solver, X)
            self.assertEqual(m._trace_P_Bq_exact(solver, X, cache), 0.0)
            self.assertTrue(np.isfinite(m._collapsed_variational_reml_Q_tau(X, .3)))

    def test_tau_statistics_are_rebuilt_after_updates(self):
        m, X = fixture()
        before = m._collapsed_variational_reml_Q_tau(X, .7)
        m.D *= 1.4
        m.y_work += .8
        m.susie_q['post_var'] *= 1.3
        after = m._collapsed_variational_reml_Q_tau(X, .7)
        self.assertNotEqual(before, after)
        P, Sigma, gram = dense_projection(m, X, .7)
        z = m.y_work - m.Z @ m.G @ m.m_sum
        expected = -.5 * (np.linalg.slogdet(Sigma)[1] + np.linalg.slogdet(gram)[1] + z @ P @ z + dense_trace(m, P))
        assert_allclose(after, expected, rtol=1e-9, atol=1e-9)


@unittest.skipUnless(os.environ.get('SCELF_BASELINE_SRC'), 'optional unmodified baseline not configured')
class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_baseline(os.environ['SCELF_BASELINE_SRC'])
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_objective_large_tau(self):
        for cov in [False, True]:
            m, X = fixture(cov=cov)
            ref, _ = fixture(cls=self.reference, cov=cov)
            for tau in [1e-8, .3, 1e4, 1e8]:
                assert_allclose(m._collapsed_variational_reml_Q_tau(X, tau), ref._collapsed_variational_reml_Q_tau(X, tau), rtol=2e-7, atol=2e-7)

    def test_credible_sets_and_sampling(self):
        for n in [1, 7]:
            for wide in [False, True]:
                p = 150 if wide else 13
                ids = np.arange(40) % n
                rng = np.random.default_rng(31)
                base = rng.normal(size=(n, 1))
                G = base + .03 * rng.normal(size=(n, p))
                if not wide and n > 1:
                    G[:, -1] = 0
                    G[0, -2] = np.nan
                args = (np.ones(40), None, None, np.eye(n)[ids], G, 2)
                m, ref = [c(*args, phi=8, seed=123) for c in [ScELF, self.reference]]
                for model in [m, ref]:
                    model.susie_q['pip_comp'][0] = np.ones(p) / p
                    model.susie_q['pip_comp'][1] = np.arange(1, p + 1) / np.arange(1, p + 1).sum()
                a, b = m.get_credible_sets(), ref.get_credible_sets()
                self.assertEqual(a.keys(), b.keys())
                if a['cs'] is not None:
                    self.assertEqual(a['cs'].keys(), b['cs'].keys())
                    for key in a['cs']:
                        np.testing.assert_array_equal(a['cs'][key], b['cs'][key])
                    assert_allclose(a['purity'], b['purity'], rtol=1e-12, atol=1e-12)
                    assert_allclose(a['coverage'], b['coverage'], atol=1e-12)
                self.assertEqual(m.rng.bit_generator.state, ref.rng.bit_generator.state)

    def test_full_fit_fixed_bounded_and_ai(self):
        for method in ['fixed', 'bounded', 'ai']:
            for cov in [False, True]:
                m, _ = fixture(cov=cov)
                ref, _ = fixture(cls=self.reference, cov=cov)
                options = {'phi': {'update': False}, 'tau': {'method': method if method != 'fixed' else 'bounded', 'tau_max': 100.0, 'trace_samples': 8}}
                for model in [m, ref]:
                    model.tau_fixed = method == 'fixed'
                    model.fit(max_outer_iter=4, tol_F=0, verbose=False, stage2_options=options)
                for key in m.susie_q:
                    assert_allclose(m.susie_q[key], ref.susie_q[key], rtol=3e-5, atol=3e-7)
                assert_allclose(m.history['F'], ref.history['F'], rtol=3e-6, atol=3e-6)
                assert_allclose(m.history['tau'], ref.history['tau'], rtol=3e-5, atol=3e-7)
                assert_allclose(m.pip, ref.pip, rtol=3e-5, atol=3e-7)
                self.assertEqual(m.tau_frozen_at_lower_bound, ref.tau_frozen_at_lower_bound)

    def test_default_tau_bounds_phi_updates_and_convergence(self):
        for seed in [42, 43, 44]:
            m, _ = fixture(n=30, cells=600, p=80, L=4, seed=seed, genotype_scale=.5, effect=1.2)
            ref, _ = fixture(cls=self.reference, n=30, cells=600, p=80, L=4, seed=seed, genotype_scale=.5, effect=1.2)
            for model in [m, ref]:
                model.fit(max_outer_iter=20, verbose=False, stage2_options={'e_step': {'ibss_sweeps': 2}})
            self.assertEqual(len(m.history['F']), len(ref.history['F']))
            assert_allclose(m.history['F'], ref.history['F'], rtol=2e-7, atol=2e-6)
            assert_allclose(m.history['phi'], ref.history['phi'], rtol=2e-6, atol=2e-7)
            assert_allclose(m.history['tau'], ref.history['tau'], rtol=2e-5, atol=2e-7)
            assert_allclose(m.get_variant_pips(False), ref.get_variant_pips(False), rtol=2e-5, atol=2e-7)
            self.assertEqual(m.sets['cs'] is None, ref.sets['cs'] is None)
            if m.sets['cs'] is not None:
                self.assertEqual(m.sets['cs'].keys(), ref.sets['cs'].keys())
                for key in m.sets['cs']:
                    np.testing.assert_array_equal(m.sets['cs'][key], ref.sets['cs'][key])

    def test_tau_lower_bound_freeze(self):
        m, X = fixture()
        ref, _ = fixture(cls=self.reference)
        options = {'tau': {'tau_min': 100., 'tau_max': 1000., 'boundary_tol': .01}}
        for model in [m, ref]:
            model._update_tau_direct(X, options)
        self.assertEqual(m.tau_frozen_at_lower_bound, ref.tau_frozen_at_lower_bound)
        self.assertTrue(m.tau_frozen_at_lower_bound)
        assert_allclose(m.tau, ref.tau, atol=1e-12)

    def test_stopping_rule(self):
        m, _ = fixture(genotype_scale=.5, effect=1.2)
        ref, _ = fixture(cls=self.reference, genotype_scale=.5, effect=1.2)
        for model in [m, ref]:
            model.fit(max_outer_iter=50, tol_F=.01, verbose=False,
                      stage2_options={'phi': {'update': False}})
        self.assertLess(len(m.history['F']), 50)
        self.assertEqual(len(m.history['F']), len(ref.history['F']))
        assert_allclose(m.history['F'], ref.history['F'], rtol=2e-7, atol=2e-7)

    def test_stage2_with_nontrivial_posterior_and_high_ld(self):
        m, _ = fixture(n=30, cells=600, p=40, L=3, genotype_scale=.5, effect=1.2)
        ref, _ = fixture(cls=self.reference, n=30, cells=600, p=40, L=3, genotype_scale=.5, effect=1.2)
        for model in [m, ref]:
            model.G[:, 1] = model.G[:, 0]
            model.G[:, -1] = 0.
            model.tau_fixed = True
            model._stage2_variational_em(True, 6, -np.inf, {'phi': {'update': False}}, verbose=False)
        self.assertGreater(float(m.susie_q['pip_comp'].max()), .2)
        self.assertGreater(float(np.abs(m.m_sum).max()), .01)
        for key in m.susie_q:
            assert_allclose(m.susie_q[key], ref.susie_q[key], rtol=2e-7, atol=2e-9)
        assert_allclose(m.history['F'], ref.history['F'], rtol=2e-7, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
