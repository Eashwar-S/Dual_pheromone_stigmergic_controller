import os
import tempfile
import unittest

try:
    import matplotlib
except ModuleNotFoundError:  # pragma: no cover - handled via skip
    matplotlib = None

if matplotlib is not None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import simulation
else:  # pragma: no cover - simplifies conditional imports
    plt = None
    FuncAnimation = None
    simulation = None


@unittest.skipIf(simulation is None, "Matplotlib is not available")
class SimulationModuleTests(unittest.TestCase):
    def test_compute_fps_rounding(self):
        self.assertEqual(simulation.compute_fps(80), 12)
        self.assertEqual(simulation.compute_fps(1000), 1)

    def test_compute_fps_invalid(self):
        with self.assertRaises(ValueError):
            simulation.compute_fps(0)

    def test_make_writer_uses_interval(self):
        writer = simulation.make_writer(200, title="Demo", artist="tester")
        self.assertEqual(writer.fps, simulation.compute_fps(200))
        self.assertEqual(writer.metadata["title"], "Demo")

    def test_run_animation_returns_animation(self):
        fig = plt.figure()
        try:
            anim = simulation.run_animation(fig, lambda _: None, interval_ms=100, frames=5)
            self.assertIsInstance(anim, FuncAnimation)
        finally:
            plt.close(fig)

    def test_frame_writer_saves_incrementally(self):
        fig = plt.figure()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                writer = simulation.FrameWriter(tmpdir, dpi=50)
                first = writer.save(fig)
                second = writer.save(fig)
                self.assertTrue(os.path.exists(first))
                self.assertTrue(os.path.exists(second))
                self.assertNotEqual(first, second)
                self.assertEqual(writer.counter, 2)
        finally:
            plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
