# import unittest

# try:
#     import numpy as np
# except ModuleNotFoundError:  # pragma: no cover
#     np = None

# try:
#     import matplotlib
# except ModuleNotFoundError:  # pragma: no cover
#     matplotlib = None

# if np is not None and matplotlib is not None:
#     matplotlib.use("Agg", force=True)
#     import centralized_approach as ca
# else:  # pragma: no cover
#     ca = None


# @unittest.skipIf(ca is None, "NumPy or Matplotlib is not available")
# class CentralizedApproachTests(unittest.TestCase):
#     def setUp(self):
#         ca.rng = np.random.default_rng(42)

#     def test_generate_unique_targets(self):
#         targets = ca.generate_unique_targets(5, 4)
#         self.assertEqual(len(targets), 4)
#         for x, y in targets:
#             self.assertTrue(0 <= x < 5)
#             self.assertTrue(0 <= y < 5)
        
#     def test_kpp_init(self):
#         W = H = 50
#         num_robots = 8
#         yy, xx = np.mgrid[0:H, 0:W]
#         points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
#         centers = ca.kpp_init(points, 8, ca.rng)
        
#         self.assertEqual(len(centers), num_robots)   
#         for x, y in centers:
#             self.assertTrue(0 <= x < W)
#             self.assertTrue(0 <= y < H)

#     def test_manhattan_path(self):
#         path = ca.manhattan_path((0, 0), (2, 2))
#         self.assertEqual(path[0], (0, 0))
#         self.assertEqual(path[-1], (2, 2))
#         self.assertTrue(all(abs(x2 - x1) + abs(y2 - y1) == 1 for (x1, y1), (x2, y2) in zip(path, path[1:])))

#     def test_mark_visible_marks_neighbors(self):
#         grid = np.zeros((3, 3), dtype=bool)
#         ca.mark_visible(grid, 1, 1)
#         expected_true = {(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)}
#         self.assertEqual({(x, y) for y in range(3) for x in range(3) if grid[y, x]}, expected_true)

#     def test_sensor_aware_path_covers_region(self):
#         mask = np.zeros((4, 4), dtype=bool)
#         mask[1, 1:3] = True
#         mask[2, 0:2] = True
#         print(mask)
#         path = ca.sensor_aware_path_for_region(mask)
#         print(path)
#         covered = {(x, y) for (x, y) in path if mask[y, x]}
#         print(covered)
#         self.assertEqual(covered, {(1, 1), (2, 1), (0, 2), (1, 2)})


# if __name__ == "__main__":  # pragma: no cover
#     unittest.main()


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
    import matplotlib.pyplot as plt
    import centralized_approach as ca
else:  # pragma: no cover
    ca = None
    plt = None


@unittest.skipIf(ca is None, "NumPy or Matplotlib is not available")
class CentralizedApproachTests(unittest.TestCase):
    def setUp(self):
        # deterministic RNG for any function that uses the module-level rng
        ca.rng = np.random.default_rng(123)

    # -----------------------------
    # Utility + setup
    # -----------------------------
    def test_generate_unique_targets(self):
        targets = ca.generate_unique_targets(5, 4)
        # correct count and in-bounds
        self.assertEqual(len(targets), 4)
        for x, y in targets:
            self.assertTrue(0 <= x < 5)
            self.assertTrue(0 <= y < 5)
        # uniqueness
        self.assertEqual(len(targets), len(set(targets)))

        # requesting too many should raise from RNG choice
        with self.assertRaises(Exception):
            ca.generate_unique_targets(2, 5)  # 4 cells only, ask 5

    def test_kpp_init(self):
        W = H = 20
        num_robots = 5
        yy, xx = np.mgrid[0:H, 0:W]
        points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
        centers = ca.kpp_init(points, num_robots, ca.rng)

        self.assertEqual(centers.shape, (num_robots, 2))
        # all centers in-bounds
        self.assertTrue(np.all(centers[:, 0] >= 0) and np.all(centers[:, 0] < W))
        self.assertTrue(np.all(centers[:, 1] >= 0) and np.all(centers[:, 1] < H))
        # should be unique selections (very high probability on grid)
        self.assertEqual(len({tuple(c) for c in centers}), num_robots)

    def test_balanced_power_diagram_assign(self):
        # symmetric 1D-like layout across x, 2 centers left/right
        W = 20; H = 1
        points = np.column_stack((np.arange(W) + 0.5, np.zeros(W) + 0.5))
        centers = np.array([[2.5, 0.5], [17.5, 0.5]], dtype=float)
        target = W // 2
        labels, lambdas = ca.balanced_power_diagram_assign(
            points, centers, target=target, iters=50, step0=0.5, decay=0.95, rng=ca.rng
        )
        # near-balanced cluster sizes is the primary goal
        sizes = np.bincount(labels, minlength=2)
        self.assertTrue(np.all(np.abs(sizes - target) <= 2))
        # lambdas may legitimately remain zero in perfectly symmetric assignments,
        # but they must at least be finite numbers
        self.assertTrue(np.all(np.isfinite(lambdas)))

    def test_lloyd_balanced(self):
        # two tight Gaussian blobs far apart
        rng = np.random.default_rng(7)
        pts_a = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(100, 2))
        pts_b = rng.normal(loc=[15.0, 15.0], scale=0.3, size=(100, 2))
        points = np.vstack([pts_a, pts_b])
        labels, centers = ca.lloyd_balanced(points, k=2,
                                            max_iters_centers=20, max_iters_assign=40,
                                            step0=0.5, decay=0.9, rng=ca.rng)
        self.assertEqual(len(labels), len(points))
        self.assertEqual(centers.shape, (2, 2))
        # centers should be near the two blob means
        means = np.array([[5.0, 5.0], [15.0, 15.0]])
        dists = np.linalg.norm(centers[:, None, :] - means[None, :, :], axis=2)
        # each center close to one of the blob means
        self.assertTrue(np.all(np.min(dists, axis=1) < 2.0))

    # -----------------------------
    # Coverage helpers
    # -----------------------------
    def test_mark_visible_marks_neighbors(self):
        grid = np.zeros((3, 3), dtype=bool)
        ca.mark_visible(grid, 1, 1)
        expected_true = {(1, 1), (1, 0), (1, 2), (0, 1), (2, 1)}
        got_true = {(x, y) for y in range(3) for x in range(3) if grid[y, x]}
        self.assertEqual(got_true, expected_true)

    def test_neighbors_von_neumann_center_and_edges(self):
        # center
        nbs = set(ca.neighbors_von_neumann(1, 1, 3, 3))
        self.assertEqual(nbs, {(1,1), (1,0), (1,2), (0,1), (2,1)})
        # corner
        nbs0 = set(ca.neighbors_von_neumann(0, 0, 3, 3))
        self.assertEqual(nbs0, {(0,0), (1,0), (0,1)})

    def test_manhattan_connect_extends_path(self):
        path = [(0, 0)]
        ca.manhattan_connect(path, 3, 2)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 2))
        # unit 4-connected steps
        self.assertTrue(all(abs(x2-x1) + abs(y2-y1) == 1 for (x1,y1),(x2,y2) in zip(path, path[1:])))

    def test_manhattan_path_monotone(self):
        p = ca.manhattan_path((2, 3), (5, 1))
        self.assertEqual(p[0], (2, 3))
        self.assertEqual(p[-1], (5, 1))
        self.assertTrue(all(abs(x2-x1) + abs(y2-y1) == 1 for (x1,y1),(x2,y2) in zip(p, p[1:])))

    # -----------------------------
    # Region sweep path (per-robot)
    # -----------------------------
    def test_sensor_aware_path_covers_region(self):
    
        # 5x5 mask
        mask = np.zeros((5, 5), dtype=bool)
        # Even row (visited): a contiguous run that the path will traverse
        mask[2, 1:4] = True     # (1,2), (2,2), (3,2)
        # Odd rows (not visited): cells vertically adjacent to the even row run
        mask[1, 2] = True       # directly above (2,2) -> should be covered
        mask[3, 2] = True       # directly below (2,2) -> should be covered

        path = ca.sensor_aware_path_for_region(mask)

        # Path is 4-connected
        self.assertTrue(all(
            abs(x2 - x1) + abs(y2 - y1) == 1
            for (x1, y1), (x2, y2) in zip(path, path[1:])
        ))

        # Simulate sensor coverage along the path
        covered = np.zeros_like(mask, dtype=bool)
        for x, y in path:
            ca.mark_visible(covered, x, y)

        # All True mask cells should be covered by the VN footprint
        trues = np.argwhere(mask)
        for (yy, xx) in trues:
            self.assertTrue(covered[yy, xx], msg=f"({xx},{yy}) not covered by sensor")

    # -----------------------------
    # Visualization helpers
    # -----------------------------
    def test_coverage_to_image_values(self):
        covered = np.zeros((2,2), dtype=bool)
        covered[0,0] = True
        img = ca.coverage_to_image(covered)
        self.assertEqual(img.shape, (2,2))
        self.assertAlmostEqual(img[0,0], 0.85, places=6)
        self.assertAlmostEqual(img[1,1], 1.0, places=6)

    def test_region_colors_shapes_and_alpha(self):
        zones = np.array([[0,1],[1,0]])
        rgba, mapping = ca.region_colors(zones, alpha=0.3)
        self.assertEqual(rgba.shape, zones.shape + (4,))
        self.assertTrue(np.allclose(rgba[...,3], 0.3))
        self.assertEqual(set(mapping.keys()), set(np.unique(zones).tolist()))

    def test_draw_voronoi_borders_plots_lines(self):
        zones = np.array([[0,1],[1,1]])
        fig, ax = plt.subplots()
        ca.draw_voronoi_borders(ax, zones)
        # should have plotted some line segments
        self.assertGreater(len(ax.lines), 0)
        plt.close(fig)

    # -----------------------------
    # Data structure: Robot
    # -----------------------------
    def test_robot_pos_and_step(self):
        r = ca.Robot(id=0, path=[(0,0),(1,0),(1,1)])
        self.assertEqual(r.pos, (0,0))
        r.step()
        self.assertEqual(r.pos, (1,0))
        r.step(); r.step()  # should stop at last
        self.assertEqual(r.pos, (1,1))
        r.step()
        self.assertEqual(r.pos, (1,1))  # still last

    # -----------------------------
    # Sensing
    # -----------------------------
    def test_discover_targets_in_vnhood(self):
        # prepare minimal globals needed by function
        ca.covered = np.zeros((5,5), dtype=bool)
        targets = {(2,2), (4,4)}
        found = set()
        ca.discover_targets_in_vnhood(3,3, targets, found)  # sees (3,3) and its VN neighbors
        # should NOT find any
        self.assertEqual(found, set())
        ca.discover_targets_in_vnhood(3,3, {(2,3)}, found)
        self.assertIn((2,3), found)
        # starting exactly on a target
        found2 = set()
        ca.discover_targets_in_vnhood(2,2, {(2,2)}, found2)
        self.assertEqual(found2, {(2,2)})

    # -----------------------------
    # Minimal integration: sim_step
    # -----------------------------
    def test_sim_step_marks_and_discovers(self):
        # Setup tiny world and one robot with a 2-cell path
        ca.covered = np.zeros((5,5), dtype=bool)
        ca.targets = {(1,0), (3,3)}
        ca.found_targets = set()
        r = ca.Robot(id=0, path=[(0,0),(1,0)])
        ca.robots = [r]

        # Before sim_step, nothing covered
        self.assertFalse(ca.covered.any())
        ca.sim_step()
        # After one sim step, both start (0,0) and next (1,0) VN-hoods marked
        # Check a few expected covered cells
        self.assertTrue(ca.covered[0,0])  # start
        self.assertTrue(ca.covered[0,1])  # end
        # Target (1,0) should be discovered by VN-hood sensing
        self.assertIn((1,0), ca.found_targets)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
