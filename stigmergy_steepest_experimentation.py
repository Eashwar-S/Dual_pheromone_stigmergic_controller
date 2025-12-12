import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Iterable
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Core utilities (ported/adapted from stigmergy.py)
# -----------------------------------------------------------------------------

@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray  # (H,W) bool, private per-robot map
    last_move: Optional[Tuple[int, int]] = None
    failed: bool = False

    def choose_move(self, pher: np.ndarray, rng: np.random.Generator,
                    bias_alpha: float, uncovered_bonus: float) -> Tuple[int, int]:
        """
        Move to the VN neighbor that maximizes discovery of cells unseen by *any* robot.
        - "Unseen by me": self.local_covered == False
        - "Unseen by anyone": pher == 0 in that region (proxy for others' coverage)
        Tie-breaks by lower pheromone to reduce redundancy.
        If surrounded (no candidate leads to any-anyone-new cells), perform a longer
        random walk to escape the pheromone basin. Never 'stay'.
        """
        H, W = pher.shape
        candidates: List[Tuple[int, int]] = []
        if self.y - 1 >= 0: candidates.append((self.x, self.y - 1))  # up
        if self.y + 1 < H:  candidates.append((self.x, self.y + 1))  # down
        if self.x - 1 >= 0: candidates.append((self.x - 1, self.y))  # left
        if self.x + 1 < W:  candidates.append((self.x + 1, self.y))  # right
        if not candidates:
            return (self.x, self.y)

        # Use the same Manhattan sensing radius you use for marking visibility.
        # No new parameter exposed; fall back to 5 if no attribute exists.
        R = int(getattr(self, "sense_radius", getattr(self, "view_radius", 5)))

        def manhattan_neighborhood(cx: int, cy: int, r: int):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) + abs(dy) <= r:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            yield nx, ny

        scored: List[Tuple[Tuple[int,int], int, int, float]] = []
        for (nx, ny) in candidates:
            nb = list(manhattan_neighborhood(nx, ny, R))

            # Newly discovered by *me* (local map)
            new_me = sum(1 for (cx, cy) in nb if not self.local_covered[cy, cx])

            # Newly discovered by *anyone* (proxy: zero pheromone means not touched)
            new_anyone = sum(1 for (cx, cy) in nb
                            if (not self.local_covered[cy, cx]) and (pher[cy, cx] == 0))

            # Pheromone load in the same region (to reduce redundancy)
            pher_sum = float(np.sum([max(0.0, pher[cy, cx]) for (cx, cy) in nb])) if nb else 0.0

            # Primary score = new_anyone (strict), secondary = new_me, tertiary = -pher_sum
            scored.append(((nx, ny), new_anyone, new_me, -pher_sum))

        # ---- Decision: prefer new-to-anyone; then new-to-me; then least pheromone ----
        max_any = max(s[1] for s in scored)
        best = [s for s in scored if s[1] == max_any]

        if len(best) > 1:
            max_me = max(s[2] for s in best)
            best = [s for s in best if s[2] == max_me]

        if len(best) > 1:
            best_val = max(s[3] for s in best)  # since we stored -pher_sum
            best = [s for s in best if s[3] == best_val]

        # If *no* candidate yields "new to anyone" (max_any == 0), we are in a basin.
        # Engage escape mode: a longer biased random walk to get out of high-pher zones.
        if max_any == 0:
            # Initialize persistent escape if not present or expired
            if not hasattr(self, "_escape_steps") or self._escape_steps <= 0:
                # pick among neighbors with *minimum* immediate pheromone to start escape
                min_edge_pher = min(max(0.0, pher[ny, nx]) for (nx, ny) in candidates)
                starters = [(nx, ny) for (nx, ny) in candidates if max(0.0, pher[ny, nx]) == min_edge_pher]

                # choose heading and set fixed-length escape (no new tunables exposed)
                idx = rng.choice(len(starters)) if len(starters) > 1 else 0
                ex, ey = starters[idx]
                self._escape_dir = (ex - self.x, ey - self.y)
                self._escape_steps = 15  # fixed internal constant

            # Try to continue in the same escape heading if possible & within bounds
            dx, dy = getattr(self, "_escape_dir", (0, 0))
            tx, ty = self.x + dx, self.y + dy
            if 0 <= tx < W and 0 <= ty < H:
                self._escape_steps -= 1
                return (tx, ty)
            else:
                # If out of bounds, reset and take the best low-pher neighbor
                self._escape_steps = 0
                min_edge_pher = min(max(0.0, pher[ny, nx]) for (nx, ny) in candidates)
                low = [(nx, ny) for (nx, ny) in candidates if max(0.0, pher[ny, nx]) == min_edge_pher]
                idx = rng.choice(len(low)) if len(low) > 1 else 0
                return low[idx]

        # Normal case: pick among 'best'
        if len(best) > 1:
            idx = rng.choice(len(best))
        else:
            idx = 0
        (nx, ny), _, _, _ = best[idx]

        # If we were in escape mode and found a new-anyone direction, cancel escape
        if hasattr(self, "_escape_steps"):
            self._escape_steps = 0
        return (nx, ny)

    def step(self, pher: np.ndarray, rng: np.random.Generator,
             bias_alpha: float, uncovered_bonus: float):
        if self.failed:
            return
        nx, ny = self.choose_move(pher, rng, bias_alpha, uncovered_bonus)
        self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny

    def deposit_pheromone(self, pher: np.ndarray, amount, r):
        """
        Deposit equal pheromone in all cells within Manhattan distance <= r
        around the robot's position (x, y).
        """
        H, W = pher.shape
        x0, y0 = int(self.x), int(self.y)

        # Collect all cells within Manhattan radius
        cells = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    cx, cy = x0 + dx, y0 + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        cells.append((cx, cy))

        # Uniform pheromone level across all neighborhood cells
        if cells:
            deposit_per_cell = amount #/ len(cells)
            for (cx, cy) in cells:
                pher[cy, cx] += deposit_per_cell


# def neighbors_vn(x: int, y: int, W: int, H: int) -> List[Tuple[int, int]]:
#     out = [(x, y)]
#     if y - 1 >= 0: out.append((x, y - 1))
#     if y + 1 < H:  out.append((x, y + 1))
#     if x - 1 >= 0: out.append((x - 1, y))
#     if x + 1 < W:  out.append((x + 1, y))
#     return out

def neighbors_vn_r(x: int, y: int, W: int, H: int, r: int = 5):
    out = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:  # VN metric (Manhattan distance)
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out.append((nx, ny))
    return out

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int, r: int = 5):
    H, W = grid_bool.shape
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    grid_bool[ny, nx] = True


def discover_vn(x: int, y: int, targets: Set[Tuple[int, int]], found: Set[Tuple[int, int]], W: int, H: int):
    for (nx, ny) in neighbors_vn_r(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))


def generate_unique_targets(grid_size: int, m: int, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)


# -----------------------------------------------------------------------------
# One headless simulation (no plotting). Supports multiple failures.
# -----------------------------------------------------------------------------

def run_simulation(
    grid_size: int,
    n_robots: int,
    n_targets: int = 30,
    max_steps: Optional[int] = None,
    target_seed: Optional[int] = None,
    robot_seed: Optional[int] = None,
    # Stigmergy params
    pher_deposit: float = 1.0,
    tau_decay: float = 600.0,
    pher_min: float = 1e-6,
    bias_alpha: float = 1.0,
    uncovered_bonus: float = 10.0,
    # Failure schedule: list of (robot_id, step). Multiple entries allowed.
    failure_schedule: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    """
    Returns a dict with:
      - steps_to_complete (int): steps until all targets discovered (or max_steps if not reached)
      - targets_found_over_time (List[int]): cumulative targets discovered by step
      - fail_map (Dict[int, List[int]]): step -> [robot_ids] that failed at that step
      - failure_schedule (List[Tuple[int, int]]): normalized schedule actually used
    """
    target_rng = np.random.default_rng(target_seed)
    robot_rng = np.random.default_rng(robot_seed)
    W = H = grid_size
    if max_steps is None:
        # generous upper bound to avoid infinite loops
        max_steps = int(W * H * 10)

    # Robots start from the center (mirrors the uploaded script default)
    robot_starting_x = W // 2
    robot_starting_y = H // 2
    pts = robot_rng.random((n_robots, 2)) * np.array([W, H])
    robots = [Robot(i,  int(pts[i, 0]), int(pts[i,1]), local_covered=np.zeros((H, W), dtype=bool))
              for i in range(n_robots)]

    # Targets and state
    targets = generate_unique_targets(W, n_targets, target_rng)
    found_targets: Set[Tuple[int, int]] = set()

    covered_global = np.zeros((H, W), dtype=bool)  # union (for logic parity)
    pher = np.zeros((H, W), dtype=float)

    # Decay factor per step
    decay_factor = math.exp(-1.0 / tau_decay)

    # Normalize failure schedule
    norm_sched: List[Tuple[int, int]] = []
    if failure_schedule:
        # Deduplicate, ignore invalid robot ids, clamp positive steps
        for (rid, st) in failure_schedule:
            if 0 <= rid < n_robots and st is not None and st >= 0:
                norm_sched.append((rid, int(st)))
        # sort by step then by rid for determinism
        norm_sched.sort(key=lambda x: (x[1], x[0]))

    # Build step -> list[rid]
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)

    # Simulation loop
    targets_found_over_time: List[int] = []
    steps = 0
    while steps < max_steps:
        # Trigger failures scheduled for this step
        if steps in fail_map:
            for rid in fail_map[steps]:
                robots[rid].failed = True

        # Evaporate pheromone
        pher *= decay_factor
        pher[pher < pher_min] = 0.0

        # Sense/cover/discover; deposit; advance
        for r in robots:
            mark_visible_bool(r.local_covered, r.x, r.y)
            mark_visible_bool(covered_global, r.x, r.y)
            discover_vn(r.x, r.y, targets, found_targets, W, H)
            # pher[r.y, r.x] += pher_deposit
            r.deposit_pheromone(pher, amount=pher_deposit, r=5)

        for r in robots:
            r.step(pher, robot_rng, bias_alpha, uncovered_bonus)

        # Post-move sensing and small extra deposit
        for r in robots:
            mark_visible_bool(r.local_covered, r.x, r.y)
            mark_visible_bool(covered_global, r.x, r.y)
            discover_vn(r.x, r.y, targets, found_targets, W, H)
            r.deposit_pheromone(pher, amount=0.3*pher_deposit, r=5)
            # pher[r.y, r.x] += 0.3 * pher_deposit

        steps += 1
        targets_found_over_time.append(len(found_targets))

        if len(found_targets) >= len(targets):
            break

    return {
        "steps_to_complete": steps,
        "targets_found_over_time": targets_found_over_time,
        "fail_map": fail_map,
        "failure_schedule": norm_sched,
    }


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------

def make_random_failure_schedule(n_robots: int, n_failures: int, rng: np.random.Generator,
                                 max_steps_hint: int) -> List[Tuple[int, int]]:
    """Sample distinct robots to fail and assign failure steps spread early-mid run."""
    n_fail = int(min(max(n_failures, 0), n_robots))
    if n_fail == 0:
        return []
    robot_ids = rng.choice(n_robots, size=n_fail, replace=False)
    # Spread failures between step 5 and ~40% of hint to leave time to observe impact
    lo = 5
    hi = max(6, int(max_steps_hint * 0.4))
    steps = rng.integers(low=lo, high=hi, size=n_fail)
    return [(int(rid), int(st)) for rid, st in zip(robot_ids, steps)]


def run_experiments(
    out_dir: str = "experiments",
    grid_sizes: Iterable[int] = (25, 50, 75, 100),
    robot_counts: Iterable[int] = (5, 10, 15),
    failure_counts: Iterable[int] = range(1, 9),
    runs_per_scenario: int = 5,
    n_targets: Iterable[int] = (20, 30, 40, 50),
    base_seed: int = 7,
    # --- added hyperparameters to record ---
    pher_deposit: float = 1.0,
    tau_decay: float = 600.0,
    pher_min: float = 1e-6,
    bias_alpha: float = 1.0,
    uncovered_bonus: float = 10.0,   # not requested as a column, but passed through
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_path / "experiments_stigmergy_steepest_with_evaporation_with_failures.xlsx"

    detailed_rows: List[Dict] = []

    # assume grid_sizes, robot_counts, failure_counts are aligned by index
    for i in range(len(grid_sizes)):
        G = grid_sizes[i]
        R = robot_counts[i]
        F = failure_counts[i]

        # fixed seed per (G,R,F) to keep failure schedule identical across runs
        schedule_seed = (base_seed * 10_000) + (G * 100) + (R * 10) + (F * 1000)
        rng_sched = np.random.default_rng(schedule_seed)

        max_steps_hint = max(G * G // max(1, R), 200)
        failure_schedule = make_random_failure_schedule(R, F, rng_sched, max_steps_hint)

        for run_idx in range(1, runs_per_scenario + 1):
            # vary the run seed so targets/walks differ per run
            run_seed = schedule_seed + run_idx
            print(int(G * G * 100))
            result = run_simulation(
                grid_size=G,
                n_robots=R,
                n_targets=n_targets[i],
                max_steps=int(G * G * 10),
                target_seed=schedule_seed,
                robot_seed=run_seed,
                failure_schedule=failure_schedule,
                # pass through the stigmergy params
                pher_deposit=pher_deposit,
                tau_decay=tau_decay,
                pher_min=pher_min,
                bias_alpha=bias_alpha,
                uncovered_bonus=uncovered_bonus,
            )

            steps_to_complete = int(result["steps_to_complete"])
            fail_ids   = [rid for rid, _ in result["failure_schedule"]]
            fail_steps = [st  for _,  st in result["failure_schedule"]]

            detailed_rows.append({
                "grid_size": G,
                "n_robots": R,
                "n_failures": F,
                "run_idx": run_idx,
                # "seed": run_seed,
                "Number of targets": n_targets[i],
                "failed_robot_ids": ";".join(map(str, fail_ids)),
                "fail_steps": ";".join(map(str, fail_steps)),
                "pher_deposit": pher_deposit,
                "tau_decay": tau_decay,
                "pher_min": pher_min,
                "bias_alpha": bias_alpha,
                "steps_to_complete": steps_to_complete
            })

    # Build pandas DataFrame
    df = pd.DataFrame(detailed_rows)

    # Summary by scenario (including hyperparams)
    group_cols = [
        "grid_size", "n_robots", "n_failures", "Number of targets",
        "pher_deposit", "tau_decay", "pher_min", "bias_alpha"
    ]
    summary = (
        df.groupby(group_cols, as_index=False)
          .agg(
              runs=("steps_to_complete", "size"),
              avg_steps=("steps_to_complete", "mean"),
              std_steps=("steps_to_complete", "std"),
          )
          .fillna({"std_steps": 0.0})
    )

    # Save to a single Excel file with two sheets
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")

        # optional: auto-fit columns
        for sheet_name, frame in [("detailed", df), ("summary", summary)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx-1, idx-1, min(max_len + 2, 60))



if __name__ == "__main__":
    # Defaults match your request:
    # - Grid sizes: 25, 50, 75, 100
    # - Robots: 5, 10, 15
    # - Failures per run: 1..8
    # - 5 runs per scenario
    run_experiments(
        out_dir="experiments",
        grid_sizes=(200, 300, 400, 500),
        robot_counts=(10, 15, 20, 25),
        failure_counts=(2, 4, 6, 8),
        runs_per_scenario=5,
        n_targets=(5, 10, 12, 15),
        base_seed=7,
        pher_deposit= 1.0,
        tau_decay= 600.0,
        pher_min= 1e-6,
        bias_alpha= 250.0,
        uncovered_bonus= 10.0,   
    )
