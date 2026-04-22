"""
Centralized 2-Robot Rendezvous Experiment
==========================================
Two robots divide the grid vertically (left half / right half) and perform
systematic boustrophedon sweeps of their respective zones searching for a
single target.

Sequence for each run:
  1. Robot 0 sweeps the left half, Robot 1 sweeps the right half.
  2. The FIRST robot to detect the target within its sensor radius stops.
  3. The OTHER robot abandons its sweep and navigates to the target via
     Manhattan path, then stops.
  4. t_targets is logged when the SECOND robot arrives at the target cell.

Robot positions (sweep start-points) vary per run via per-run seeds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.geometry import manhattan_path
from src.centralized.partitioning import lloyd_balanced
from src.centralized.path_planning import sensor_aware_path_for_region
from src.experimentation import config


# ---------------------------------------------------------------------------
# Config override: only 2 robots, 1 target. Can be changed in config.py by
# adding a RENDEZVOUS_N_ROBOTS / RENDEZVOUS_N_TARGETS entry, but we derive
# them from the per-scenario counts via a cap of 2 and 1.
# ---------------------------------------------------------------------------
N_ROBOTS = 2   # exactly two robots for the rendezvous scenario
N_TARGETS = 1  # exactly one target per run


@dataclass
class RobotState:
    """Minimal robot state for the centralized rendezvous experiment."""
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0
    failed: bool = False
    stopped: bool = False      # True once this robot has found/reached the target
    found_target: bool = False # True if THIS robot was first to detect

    @property
    def pos(self) -> Tuple[int, int]:
        return self.path[self.idx]

    @property
    def x(self) -> int:
        return self.path[self.idx][0]

    @property
    def y(self) -> int:
        return self.path[self.idx][1]

    def step(self):
        """Advance one step along path unless stopped or at end."""
        if self.stopped or self.failed:
            return
        if self.idx < len(self.path) - 1:
            self.idx += 1

    def redirect_to(self, target: Tuple[int, int]):
        """Replace remaining path with Manhattan route to target."""
        nav = manhattan_path(self.pos, target)
        # Replace from current position onward
        self.path = self.path[:self.idx + 1] + nav[1:]


def run_simulation(grid_size: int, rng_seed: int,
                   robot_radius: int) -> Dict[str, object]:
    """
    Run one centralized 2-robot rendezvous simulation.

    rng_layout (seeded with rng_seed) generates positions + target — identical
    sequence to stigmergy_rendezvous_experiment for the same rng_seed.
    rng_algo (seeded with rng_seed + 1_000_000) drives Lloyd's partitioning.
    """
    # Layout RNG: positions then target — must match stigmergy call order exactly
    rng_layout = np.random.default_rng(rng_seed)
    # Algorithm RNG: partitioning (consumes unpredictable draws, kept separate)
    rng_algo = np.random.default_rng(rng_seed + 1_000_000)
    W = H = grid_size
    max_horizon = config.calculate_horizon(grid_size, N_ROBOTS)

    # --- Partition grid into 2 balanced halves via Lloyd's ---
    MAX_ITERS_ASSIGN = 30
    MAX_ITERS_CENTERS = 10
    LAMBDA_STEP0 = 0.1
    LAMBDA_DECAY = 0.1

    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    labels, centers = lloyd_balanced(points, N_ROBOTS, MAX_ITERS_CENTERS,
                                     MAX_ITERS_ASSIGN, LAMBDA_STEP0,
                                     LAMBDA_DECAY, rng_algo)
    zones = labels.reshape(H, W)
    masks = [(zones == i) for i in range(N_ROBOTS)]

    # --- Build boustrophedon sweep paths for each zone ---
    sweeping_paths: List[List[Tuple[int, int]]] = []
    for i in range(N_ROBOTS):
        p = sensor_aware_path_for_region(masks[i], robot_radius=robot_radius)
        if not p:
            cx, cy = centers[i]
            px = int(np.clip(round(cx - 0.5), 0, W - 1))
            py = int(np.clip(round(cy - 0.5), 0, H - 1))
            p = [(px, py)]
        sweeping_paths.append(p)

    # --- Random start positions (rng_layout: same as stigmergy for same seed) ---
    start_positions = config.generate_robot_positions(grid_size, N_ROBOTS, rng_layout)

    # Full path: random start → navigate to sweep start → sweep
    full_paths: List[List[Tuple[int, int]]] = []
    for i in range(N_ROBOTS):
        sp = sweeping_paths[i]
        nav = manhattan_path(start_positions[i], sp[0])
        full_paths.append(nav[:-1] + sp)

    robots = [RobotState(i, full_paths[i]) for i in range(N_ROBOTS)]

    # --- Single target (rng_layout: same as stigmergy for same seed) ---
    targets = generate_unique_targets(grid_size, N_TARGETS, rng_layout)
    target_pos: Tuple[int, int] = next(iter(targets))

    covered = np.zeros((H, W), dtype=int)

    # Rendezvous state
    first_finder: Optional[int] = None   # id of robot that detected target first
    t_first_detect = max_horizon         # step when first robot detects target
    t_targets = max_horizon              # step when second robot arrives at target

    steps = 0
    while steps < max_horizon:
        # --- Sense at current positions ---
        for r in robots:
            if not r.stopped and not r.failed:
                mark_visible(covered, r.x, r.y, robot_radius)

                # Check if this robot detects the target
                if first_finder is None:
                    tx, ty = target_pos
                    if abs(r.x - tx) + abs(r.y - ty) <= robot_radius:
                        # First detection event
                        first_finder = r.id
                        t_first_detect = steps
                        r.stopped = True     # finder stops here
                        r.found_target = True

                        # Redirect the other robot toward the target
                        other_id = 1 - r.id
                        other = robots[other_id]
                        if not other.failed:
                            other.redirect_to(target_pos)

        # --- Move non-stopped robots ---
        for r in robots:
            if not r.stopped and not r.failed:
                r.step()

        # --- Post-move sense ---
        for r in robots:
            if not r.stopped and not r.failed:
                mark_visible(covered, r.x, r.y, robot_radius)

        steps += 1

        # --- Check rendezvous: second robot reaches target cell ---
        if first_finder is not None and t_targets == max_horizon:
            other_id = 1 - first_finder
            other = robots[other_id]
            if not other.failed and other.pos == target_pos:
                t_targets = steps
                break

    # --- Build metrics ---
    current_coverage = np.sum(covered > 0)
    total_cells = H * W
    percent_coverage = (current_coverage / total_cells) * 100.0
    n_targets_found = 1 if first_finder is not None else 0

    return {
        "grid_size": grid_size,
        "n_robots": N_ROBOTS,
        "n_targets": N_TARGETS,
        "n_failures": 0,
        "n_targets_found": n_targets_found,
        "t_first_detect": t_first_detect,
        "t_targets": t_targets,        # rendezvous complete (both at target)
        "percent_coverage": percent_coverage,
    }


def run_experiments():
    """Run all grid-size scenarios, 5 runs each with varying robot positions."""
    output_path = config.get_output_path("centralized_rendezvous")
    xlsx_path = output_path / "results.xlsx"

    rows: List[Dict[str, object]] = []
    configs_list = config.get_experiment_configs()
    base_seed = 42
    # for grid_size, _n_robots, _n_targets, _n_failures in configs_list:
        

    for run_idx in range(1, 20):#config.RUNS_PER_SCENARIO + 1):
        # Different seed per run → different robot start positions & target location
        run_seed = base_seed + run_idx

        print(f"Running: grid={100}, robots={N_ROBOTS}, targets={N_TARGETS}, run={run_idx}")

        result = run_simulation(
            grid_size=100,
            rng_seed=run_seed,
            robot_radius=config.ROBOT_RADIUS
        )
        print(f"  t_first_detect: {result['t_first_detect']}, "
                f"t_targets (rendezvous): {result['t_targets']}, "
                f"percent_coverage: {result['percent_coverage']:.2f}")

        rows.append(result)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["grid_size", "n_robots", "n_targets"], as_index=False)
          .agg(
              runs=("t_targets", "size"),
              avg_n_targets_found=("n_targets_found", "mean"),
              avg_t_first_detect=("t_first_detect", "mean"),
              avg_t_targets=("t_targets", "mean"),
              avg_percent_coverage=("percent_coverage", "mean"),
          )
    )

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")

        for sheet_name, frame in [("detailed", df), ("summary", summary)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx - 1, idx - 1, min(max_len + 2, 60))

    print(f"\nResults saved to: {xlsx_path}")


if __name__ == "__main__":
    run_experiments()
