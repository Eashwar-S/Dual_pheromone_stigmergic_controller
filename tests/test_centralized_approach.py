import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None

try:
    import matplotlib
except ModuleNotFoundError:  # pragma: no cover
    matplotlib = None

if np is not None and matplotlib is not None:
    matplotlib.use("Agg", force=True)
    import centralized_approach as ca
else:  # pragma: no cover
    ca = None


@unittest.skipIf(ca is None, "NumPy or Matplotlib is not available")
class CentralizedApproachTests(unittest.TestCase):
    def setUp(self):
        ca.rng = np.random.default_rng(42)

    def test_generate_unique_targets(self):
        targets = ca.generate_unique_targets(5, 4)
        self.assertEqual(len(targets), 4)
        for x, y in targets:
            self.assertTrue(0 <= x < 5)
            self.assertTrue(0 <= y < 5)

    def test_manhattan_path(self):
        path = ca.manhattan_path((0, 0), (2, 2))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 2))
        self.assertTrue(all(abs(x2 - x1) + abs(y2 - y1) == 1 for (x1, y1), (x2, y2) in zip(path, path[1:])))

    def test_mark_visible_marks_neighbors(self):
        grid = np.zeros((3, 3), dtype=bool)
        ca.mark_visible(grid, 1, 1)
        expected_true = {(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)}
        self.assertEqual({(x, y) for y in range(3) for x in range(3) if grid[y, x]}, expected_true)

    def test_sensor_aware_path_covers_region(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1:3] = True
        mask[2, 0:2] = True
        path = ca.sensor_aware_path_for_region(mask)
        covered = {(x, y) for (x, y) in path if mask[y, x]}
        self.assertEqual(covered, {(1, 1), (2, 1), (0, 2), (1, 2)})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
