"""Numerical checks for the public PIP colocalization feature."""
import unittest
import numpy as np
from scelf import ColocPIPResult, coloc_pip


class ColocTests(unittest.TestCase):
    def test_matches_analytical_product_weights(self):
        for x, y in [([.7, .2, .1], [.6, .35, .05]), ([.9, 0, .01], [0, .7, .1]), ([1.0], [1.0])]:
            x, y = np.maximum(x, 1e-16), np.maximum(y, 1e-16)
            shared = np.dot(x, y)
            distinct = max(0, x.sum() * y.sum() - shared)
            expected = 1e-7 * shared / (1e-7 * shared + 1e-8 * distinct)
            result = coloc_pip(x, y)
            self.assertIsInstance(result, ColocPIPResult)
            self.assertAlmostEqual(result.h4, expected, places=13)
            self.assertAlmostEqual(result.h3 + result.h4, 1, places=13)
            self.assertEqual(result.score, result.h4)

    def test_permutation_and_input_symmetry(self):
        x, y = np.array([.6, .2, .1]), np.array([.5, .1, .3])
        result = coloc_pip(x, y, p1=.002, p2=.003)
        self.assertAlmostEqual(result.h4, coloc_pip(x[::-1], y[::-1], p1=.002, p2=.003).h4)
        self.assertAlmostEqual(result.h4, coloc_pip(y, x, p1=.003, p2=.002).h4)

    def test_invalid_input_is_rejected(self):
        for x, y in [([], []), ([.2], [.2, .3]), ([[.5]], [[.5]]), ([np.nan], [.2]), ([1.01], [.2]), ([-.1], [.2]), ([0], [.5])]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                coloc_pip(x, y)
        for kwargs in [dict(p1=0), dict(p12=np.inf), dict(pseudocount=0)]:
            with self.assertRaises(ValueError):
                coloc_pip([.5], [.6], **kwargs)


if __name__ == "__main__":
    unittest.main()
