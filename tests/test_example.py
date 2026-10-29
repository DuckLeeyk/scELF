"""Check compact cell-to-donor storage and the example's single-load flow."""

from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("scelf_bundled_example", ROOT / "example/run_b_memory_lst1.py")
example = module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class ExampleTests(unittest.TestCase):
    def test_mapping_round_trip(self):
        indices = np.array([2, 0, 2, 1], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "donors.npy"
            np.save(path, indices, allow_pickle=False)
            Z = example.read_design_matrix(path, 3)
        np.testing.assert_array_equal(Z, np.eye(3, dtype=np.uint8)[indices])
        self.assertEqual(Z.dtype, np.uint8)

    def test_invalid_mapping_is_rejected(self):
        invalid = [np.array([]), np.array([[0, 1]]), np.array([0.5]),
                   np.array([-1, 0]), np.array([0, 3]), np.array([True, False])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "donors.npy"
            for values in invalid:
                with self.subTest(values=values):
                    np.save(path, values, allow_pickle=False)
                    with self.assertRaises(ValueError):
                        example.read_design_matrix(path, 3)
            np.save(path, np.array([0]), allow_pickle=False)
            with self.assertRaises(ValueError):
                example.read_design_matrix(path, 0)

    def test_bundled_data_alignment_and_storage(self):
        y, Xc, Xd, Z, G, variants = example.load_example_data()
        indices = np.load(example.EXAMPLE_DIR / "cell_donor_index_B_memory.npy", allow_pickle=False)
        self.assertEqual(Z.shape, (y.size, G.shape[0]))
        self.assertEqual(Xc.shape[0], y.size)
        self.assertEqual(Xd.shape[0], G.shape[0])
        self.assertEqual(variants.size, G.shape[1])
        np.testing.assert_array_equal(Z.argmax(axis=1), indices)
        np.testing.assert_array_equal(Z.sum(axis=1), np.ones(y.size))
        self.assertTrue(np.isfinite(G).all())
        np.testing.assert_allclose(G.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(G.std(axis=0), 1, atol=1e-10)
        self.assertLess((example.EXAMPLE_DIR / "cell_donor_index_B_memory.npy").stat().st_size, 100_000)

    def test_main_loads_data_once(self):
        data = (np.array([1.]), np.ones((1, 1)), np.ones((1, 1)),
                np.ones((1, 1)), np.ones((1, 1)), np.array(["v1"]))
        model = SimpleNamespace(backend="numba", pip=np.array([0.5]), sets={})
        timing = {"fit_seconds": 0.1, "stage1_seconds": 0.02,
                  "stage2_seconds": 0.07, "outer_iterations": 1}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(example, "load_example_data", return_value=data) as loader, \
                patch.object(example, "fit_once", return_value=(model, timing)), \
                patch.object(example, "backend_info", return_value={"cpp_available": False}), \
                redirect_stdout(io.StringIO()):
            example.main(["--backend", "numba", "--quiet", "--output-dir", directory])
            loader.assert_called_once_with()
            self.assertTrue((Path(directory) / "B_memory_LST1_pip.csv").is_file())


if __name__ == "__main__":
    unittest.main()
