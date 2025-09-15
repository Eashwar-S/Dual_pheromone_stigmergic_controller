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
    import stigmergy
else:  # pragma: no cover
    stigmergy = None


@unittest.skipIf(stigmergy is None, "NumPy or Matplotlib is not available")
class StigmergyModuleTests(unittest.TestCase):
    def setUp(self):
        stigmergy.rng = np.random.default_rng(24)
        stigmergy.BIAS_ALPHA = getattr(stigmergy, "BIAS_ALPHA", 2.5)
        stigmergy.UNCOVERED_BONUS = getattr(stigmergy, "UNCOVERED_BONUS", 10.0)

    def test_generate_unique_targets(self):
        targets = stigmergy.generate_unique_targets(6, 5)
        self.assertEqual(len(targets), 5)
        for x, y in targets:
            self.assertTrue(0 <= x < 6)
            self.assertTrue(0 <= y < 6)

    def test_mark_visible_bool_marks_neighbors(self):
        grid = np.zeros((3, 3), dtype=bool)
        stigmergy.mark_visible_bool(grid, 1, 1)
        expected = {(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)}
        self.assertEqual({(x, y) for y in range(3) for x in range(3) if grid[y, x]}, expected)

    def test_discover_vn_discovers_targets(self):
        targets = {(1, 1), (0, 0)}
        found = set()
        stigmergy.discover_vn(1, 1, targets, found, 3, 3)
        self.assertIn((1, 1), found)
        self.assertIn((0, 0), found)

    def test_robot_choose_move_prefers_uncovered(self):
        robot = stigmergy.Robot(0, 1, 1, local_covered=np.ones((3, 3), dtype=bool))
        robot.local_covered[1, 2] = False  # only east is uncovered
        pher = np.zeros((3, 3), dtype=float)
        move = robot.choose_move(pher)
        self.assertEqual(move, (2, 1))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
