"""Native kernel, backend selection, and cross-backend numerical tests."""
import copy
from functools import partial
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits
from test_donor_compression import ScELF, fixture
from scelf import backend_info
from scelf import backend as selection
from scelf.susie import _loglik_core, optimize_prior_variance, SuSiE


class BackendSelectionTests(unittest.TestCase):
    def test_explicit_numba_and_invalid_selection(self):
        m, _ = fixture(partial(ScELF, backend='numba'))
        self.assertEqual(m.backend, 'numba')
        with self.assertRaises(ValueError):
            fixture(partial(ScELF, backend='invalid'))

    def test_unavailable_extension_fallback_and_required_failure(self):
        with patch.multiple(selection, _checked=True, _native=None, _native_error='test missing extension'):
            m, _ = fixture()
            self.assertEqual(m.backend, 'numba')
            self.assertFalse(backend_info()['cpp_available'])
            with self.assertRaisesRegex(ImportError, 'test missing extension'):
                fixture(partial(ScELF, backend='cpp'))


@unittest.skipUnless(backend_info()['cpp_available'], 'optional native extension is not built')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = selection._load_cpp()
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_auto_and_deepcopy(self):
        model, _ = fixture()
        self.assertEqual(model.backend, 'cpp')
        clone = copy.deepcopy(model)
        assert_allclose(clone.G, model.G)

    def test_diagonal_and_trace_shapes_types_and_empty_dimensions(self):
        model, X = fixture(partial(ScELF, backend='cpp'))
        solver = model._make_exact_solver()
        cache = model._make_projection_cache(solver, X)
        T, A = model._cpp_projection_arrays(cache)
        u = solver.w_donor * solver.c_coef
        with self.assertRaises(ValueError):
            self.native.diag_gpg(model.G, u[:-1], T, A)
        with self.assertRaises(TypeError):
            self.native.diag_gpg(model.G.astype(np.float32), u, T, A)
        with self.assertRaises(TypeError):
            self.native.diag_gpg(np.array(model.G, order='C'), u, T, A)
        for n, p in [(0, 4), (3, 0)]:
            diagonal = self.native.diag_gpg(np.empty((n, p), order='F'), np.ones(n), np.empty((n, 0), order='F'), np.empty((0, 0), order='F'))
            assert_allclose(diagonal, np.full(p, 1e-12))

    def test_readonly_and_overlapping_output_rejected(self):
        model, X = fixture(partial(ScELF, backend='cpp'))
        solver = model._make_exact_solver()
        cache = model._make_projection_cache(solver, X)
        T, A = model._cpp_projection_arrays(cache)
        q = model.susie_q
        args = [model.G, solver.w_donor * solver.c_coef, T, A, np.ones(model.n), np.ones(model.p),
                q['pip_comp'], q['post_mean'], q['post_var'], q['m_comp'], q['prior_var'], model.m_sum]
        q['post_mean'].flags.writeable = False
        with self.assertRaisesRegex(ValueError, 'writable'):
            self.native.ibss_sweep(*args)
        q['post_mean'].flags.writeable = True
        args[7] = args[6]
        before = model.m_sum.copy()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.native.ibss_sweep(*args)
        assert_allclose(model.m_sum, before)

    def test_prior_likelihood_invalid_columns_and_wide_variances(self):
        beta = np.array([.2, -.8, 1.7, 0., 4., 2.])
        shat = np.array([.01, 1., 20., np.inf, 0., np.nan])
        prior = np.array([.1, .2, .3, 0., .15, .25])
        prepared = self.native.PriorLikelihood(beta, shat, prior)
        for V in [0., 1e-14, .2, 100., 1e12]:
            assert_allclose(prepared.value(V), _loglik_core(V, beta, shat, prior), rtol=1e-12, atol=1e-12)
        # Constructor copies sufficient statistics; source mutation cannot stale it.
        previous = prepared.value(.2)
        beta[:] = 9
        self.assertEqual(previous, prepared.value(.2))
        with self.assertRaises(ValueError):
            self.native.PriorLikelihood(beta, shat[:-1], prior)
        with self.assertRaises(ValueError):
            self.native.PriorLikelihood(np.empty(0), np.empty(0), np.empty(0))

    def test_prior_optimizer_methods_and_null_choice(self):
        rng = np.random.default_rng(53)
        beta = rng.normal(size=100)
        variance = rng.uniform(.1, 2., 100)
        prior = np.full(100, .01)
        for method in ['optim', 'uniroot', 'none', 'EM']:
            kwargs = {'V_init': .2, 'alpha': prior, 'post_mean2': beta**2, 'check_null_threshold': .01}
            a = optimize_prior_variance(method, beta, variance, prior, backend='numba', **kwargs)
            b = optimize_prior_variance(method, beta, variance, prior, backend='cpp', **kwargs)
            assert_allclose(a, b, rtol=2e-7, atol=2e-9)
        beta[:] = 0
        self.assertEqual(optimize_prior_variance('optim', beta, variance, prior, V_init=.2, backend='cpp'), 0.)

    def test_multiple_sweeps_nontrivial_posterior(self):
        for cov in [False, True]:
            for update in [False, True]:
                models = [fixture(partial(ScELF, backend=b), cov=cov)[0] for b in ['numba', 'cpp']]
                for _ in range(4):
                    for model in models:
                        X = model._build_design_X() if cov else np.zeros((model.M, 0))
                        model._ibss_full_cycle_collapsed(X, update)
                    for key in models[0].susie_q:
                        assert_allclose(models[0].susie_q[key], models[1].susie_q[key], rtol=2e-9, atol=2e-10)

    def test_full_fit_and_standalone_susie(self):
        models = [fixture(partial(ScELF, backend=b), n=30, cells=600, p=40, L=3, genotype_scale=.5, effect=1.2)[0] for b in ['numba', 'cpp']]
        for model in models:
            model.fit(max_outer_iter=6, tol_F=-np.inf, verbose=False, stage2_options={'tau': {'tau_max': 100.}})
        for key in models[0].susie_q:
            assert_allclose(models[0].susie_q[key], models[1].susie_q[key], rtol=2e-6, atol=2e-8)
        assert_allclose(models[0].history['F'], models[1].history['F'], rtol=2e-7, atol=2e-6)
        rng = np.random.default_rng(7)
        G = .5 * rng.normal(size=(100, 30))
        y = 2. * G[:, 0] + rng.normal(scale=.1, size=100)
        fits = [SuSiE(backend=b) for b in ['numba', 'cpp']]
        for model in fits:
            model.fit(G, y, L=3, verbose=False)
        self.assertGreater(float(fits[0].alpha.max()), .5)
        assert_allclose(fits[0].alpha, fits[1].alpha, rtol=2e-6, atol=2e-8)
        assert_allclose(fits[0].V, fits[1].V, rtol=2e-6, atol=2e-8)


if __name__ == '__main__':
    unittest.main()
