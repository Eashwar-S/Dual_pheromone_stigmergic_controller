import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict, Iterable
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Core utilities (ported/adapted; changes tagged with "CHANGED"/"NEW")
# ---------------------------------------------------------------------

def neighbors_vn(x: int, y: int, W: int, H: int) -> List[Tuple[int, int]]:
    out = [(x, y)]
    if y - 1 >= 0: out.append((x, y - 1))
    if y + 1 < H:  out.append((x, y + 1))
    if x - 1 >= 0: out.append((x - 1, y))
    if x + 1 < W:  out.append((x + 1, y))
    return out

def neighbors_vn_4(x: int, y: int, W: int, H: int) -> List[Tuple[int, int]]:
    """4-neighborhood (no center)."""
    out: List[Tuple[int, int]] = []
    if y - 1 >= 0: out.append((x, y - 1))
    if y + 1 < H:  out.append((x, y + 1))
    if x - 1 >= 0: out.append((x - 1, y))
    if x + 1 < W:  out.append((x + 1, y))
    return out

def surrounded_by_pheromone(x: int, y: int, pher: np.ndarray) -> bool:
    """Return True if all 4 VN neighbors have >0 pheromone."""
    H, W = pher.shape
    n4 = neighbors_vn_4(x, y, W, H)
    if not n4:
        return False
    for (nx, ny) in n4:
        if pher[ny, nx] <= 0.0:
            return False
    return True

def mark_and_check_new(grid_bool: np.ndarray, x: int, y: int) -> bool:
    """
    Mark VN neighborhood (incl center) and return True if
    at least one newly covered cell was found.
    """
    H, W = grid_bool.shape
    new_found = False
    # center
    if not grid_bool[y, x]:
        new_found = True
        grid_bool[y, x] = True
    # 4-neighbors
    if y - 1 >= 0 and not grid_bool[y - 1, x]:
        new_found = True
        grid_bool[y - 1, x] = True
    if y + 1 < H and not grid_bool[y + 1, x]:
        new_found = True
        grid_bool[y + 1, x] = True
    if x - 1 >= 0 and not grid_bool[y, x - 1]:
        new_found = True
        grid_bool[y, x - 1] = True
    if x + 1 < W and not grid_bool[y, x + 1]:
        new_found = True
        grid_bool[y, x + 1] = True
    return new_found

def discover_vn(x: int, y: int, targets: Set[Tuple[int, int]], found: Set[Tuple[int, int]], W: int, H: int):
    for (nx, ny) in neighbors_vn(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

def generate_unique_targets(grid_size: int, m: int, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray  # (H,W) bool
    last_move: Optional[Tuple[int, int]] = None
    failed: bool = False
    # ----------------- NEW (anti-stagnation state) -----------------
    stagnation_steps: int = 0
    random_walk_remaining: int = 0

    # ------------------- CHANGED: now supports uniform walk -------------
    def choose_move(self, pher: np.ndarray, rng: np.random.Generator,
                    bias_alpha: float, uncovered_bonus: float,
                    uniform_random: bool = False) -> Tuple[int, int]:
        """Pick next VN cell (no 'stay'). If uniform_random=True, ignore pheromone."""
        H, W = pher.shape
        candidates: List[Tuple[int, int]] = []
        if self.y - 1 >= 0: candidates.append((self.x, self.y - 1))
        if self.y + 1 < H:  candidates.append((self.x, self.y + 1))
        if self.x - 1 >= 0: candidates.append((self.x - 1, self.y))
        if self.x + 1 < W:  candidates.append((self.x + 1, self.y))
        if not candidates:
            return self.x, self.y

        if uniform_random:
            idx = rng.integers(len(candidates))
            return candidates[int(idx)]

        # Original biased choice (avoid pher, prefer locally-uncovered)
        uncov = [(nx, ny) for (nx, ny) in candidates if not self.local_covered[ny, nx]]
        pool = uncov if len(uncov) > 0 else candidates
        weights = []
        for (nx, ny) in pool:
            p = max(pher[ny, nx], 0.0)
            desirability = math.exp(-bias_alpha * (p / (1.0 + p)))
            if not self.local_covered[ny, nx]:
                desirability *= uncovered_bonus
            weights.append(desirability)
        w = np.array(weights, dtype=float)
        if np.all(w <= 0):
            w = np.ones_like(w)
        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[int(idx)]

    def step(self, pher: np.ndarray, rng: np.random.Generator,
             bias_alpha: float, uncovered_bonus: float):
        if self.failed:
            return
        # If on a random-walk interval, ignore pheromone completely
        uniform = self.random_walk_remaining > 0
        nx, ny = self.choose_move(pher, rng, bias_alpha, uncovered_bonus, uniform_random=uniform)
        self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        if self.random_walk_remaining > 0:
            self.random_walk_remaining -= 1

# ---------------------------------------------------------------------
# One headless simulation (no plotting). Supports multiple failures.
# ---------------------------------------------------------------------

def run_simulation(
    grid_size: int,
    n_robots: int,
    n_targets: int = 30,
    max_steps: Optional[int] = None,
    target_seed: Optional[int] = None,
    robot_seed: Optional[int] = None,
    # Stigmergy params (kept for compatibility)
    pher_deposit: float = 1.0,
    tau_decay: float = 600.0,   # will be ignored (no evaporation)
    pher_min: float = 1e-6,     # threshold kept but unused in decay
    bias_alpha: float = 1.0,
    uncovered_bonus: float = 10.0,
    # NEW anti-stagnation knobs (fixed across the experiment)
    stagnation_x: int = 15,     # if no new ground for x consecutive steps...
    randomwalk_y: int = 10,     # ... do uniform random walk for y steps
    # Failure schedule: list of (robot_id, step)
    failure_schedule: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    """
    Returns:
      steps_to_complete (int)
      targets_found_over_time (List[int])
      fail_map (Dict[int, List[int]])
      failure_schedule (List[Tuple[int, int]])
    """
    target_rng = np.random.default_rng(target_seed)
    robot_rng = np.random.default_rng(robot_seed)
    W = H = grid_size
    if max_steps is None:
        max_steps = int(W * H * 10)

    # Robots start from the center
    sx, sy = W // 2, H // 2
    robots = [Robot(i, sx, sy, local_covered=np.zeros((H, W), dtype=bool))
              for i in range(n_robots)]

    # Targets & world state
    targets = generate_unique_targets(W, n_targets, target_rng)
    found_targets: Set[Tuple[int, int]] = set()
    covered_global = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)

    # ------------------- CHANGED: disable evaporation -------------------
    decay_factor = 1.0  # (No evaporation) previously: exp(-1/tau_decay)

    # Normalize failure schedule → step -> [rids]
    norm_sched: List[Tuple[int, int]] = []
    if failure_schedule:
        for (rid, st) in failure_schedule:
            if 0 <= rid < n_robots and st is not None and st >= 0:
                norm_sched.append((rid, int(st)))
        norm_sched.sort(key=lambda x: (x[1], x[0]))
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)

    targets_found_over_time: List[int] = []
    steps = 0
    while steps < max_steps:
        # Trigger failures for this step
        if steps in fail_map:
            for rid in fail_map[steps]:
                robots[rid].failed = True

        # ------------------- CHANGED: no evaporation ---------------------
        pher *= decay_factor
        # (keep a safety clamp if you like; it's moot with factor=1.0)
        pher[pher < 0.0] = 0.0

        # Sense/cover/discover; conditional deposit
        any_new_covered_this_step = [False] * n_robots
        for r in robots:
            # mark local & global; capture if new ground was covered
            new_local = mark_and_check_new(r.local_covered, r.x, r.y)
            new_global = mark_and_check_new(covered_global, r.x, r.y)
            any_new_covered_this_step[r.id] = (new_local or new_global)
            discover_vn(r.x, r.y, targets, found_targets, W, H)

            # ------------------- NEW: gated deposit ----------------------
            # If surrounded by pheromone in VN, skip deposit
            if not surrounded_by_pheromone(r.x, r.y, pher):
                pher[r.y, r.x] += pher_deposit

        # Update stagnation counters and trigger random walks
        for r in robots:
            if r.failed:
                continue
            if any_new_covered_this_step[r.id]:
                r.stagnation_steps = 0
            else:
                r.stagnation_steps += 1
                if r.stagnation_steps >= stagnation_x and r.random_walk_remaining == 0:
                    r.random_walk_remaining = randomwalk_y
                    r.stagnation_steps = 0  # reset after triggering the walk

        # Move
        for r in robots:
            r.step(pher, robot_rng, bias_alpha, uncovered_bonus)

        # Post-move sensing; conditional deposit
        for r in robots:
            new_local = mark_and_check_new(r.local_covered, r.x, r.y)
            new_global = mark_and_check_new(covered_global, r.x, r.y)
            any_new_covered = (new_local or new_global)
            if any_new_covered:
                r.stagnation_steps = 0  # also break stagnation if post-move found new

            discover_vn(r.x, r.y, targets, found_targets, W, H)

            # ------------------- NEW: gated deposit (again) --------------
            if not surrounded_by_pheromone(r.x, r.y, pher):
                # small post-move deposit as in original; keep factor 0.3
                pher[r.y, r.x] += 0.3 * pher_deposit

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


# ---------------------------------------------------------------------
# Experiment driver (kept structure; adds x,y columns) 
# ---------------------------------------------------------------------

def make_random_failure_schedule(n_robots: int, n_failures: int, rng: np.random.Generator,
                                 max_steps_hint: int) -> List[Tuple[int, int]]:
    n_fail = int(min(max(n_failures, 0), n_robots))
    if n_fail == 0:
        return []
    robot_ids = rng.choice(n_robots, size=n_fail, replace=False)
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
    # --- existing hyperparams ---
    pher_deposit: float = 1.0,
    tau_decay: float = 600.0,   # ignored (no evap)
    pher_min: float = 1e-6,
    bias_alpha: float = 1.0,
    uncovered_bonus: float = 10.0,
    # --- NEW: fixed across the whole experiment; saved to XLSX ---
    stagnation_x: int = 15,
    randomwalk_y: int = 10,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_path / f"experiments_stigmergy_no_evaporation_{stagnation_x}_{randomwalk_y}.xlsx"

    detailed_rows: List[Dict] = []

    for i in range(len(grid_sizes)):
        G = grid_sizes[i]
        R = robot_counts[i]
        F = failure_counts[i]

        # fixed schedule seed per scenario (keep failures identical across runs)
        schedule_seed = (base_seed * 10_000) + (G * 100) + (R * 10) + (F * 1000)
        rng_sched = np.random.default_rng(schedule_seed)

        max_steps_hint = max(G * G // max(1, R), 200)
        failure_schedule = make_random_failure_schedule(R, F, rng_sched, max_steps_hint)

        for run_idx in range(1, runs_per_scenario + 1):
            # vary per-run seed (so behavior differs across runs)
            run_seed = schedule_seed + run_idx
            # print(int(G * G * 100))  # (optional) noisy

            result = run_simulation(
                grid_size=G,
                n_robots=R,
                n_targets=n_targets[i],
                max_steps=int(G * G * 10),
                target_seed=schedule_seed,   # keep targets fixed per scenario
                robot_seed=run_seed,         # let robot behavior vary per run
                failure_schedule=failure_schedule,
                # pass through parameters
                pher_deposit=pher_deposit,
                tau_decay=tau_decay,     # ignored (no evap)
                pher_min=pher_min,
                bias_alpha=bias_alpha,
                uncovered_bonus=uncovered_bonus,
                stagnation_x=stagnation_x,    # NEW
                randomwalk_y=randomwalk_y,    # NEW
            )

            steps_to_complete = int(result["steps_to_complete"])
            fail_ids   = [rid for rid, _ in result["failure_schedule"]]
            fail_steps = [st  for _,  st in result["failure_schedule"]]

            # ------------- CHANGED: add stagnation_x, randomwalk_y cols -------------
            detailed_rows.append({
                "grid_size": G,
                "n_robots": R,
                "n_failures": F,
                "run_idx": run_idx,
                # "seed": run_seed,  # keep your previous choice/comment
                "Number of targets": n_targets[i],
                "failed_robot_ids": ";".join(map(str, fail_ids)),
                "fail_steps": ";".join(map(str, fail_steps)),
                # "pher_deposit": pher_deposit,
                # "tau_decay": tau_decay,
                # "pher_min": pher_min,
                # "bias_alpha": bias_alpha,
                "stagnation_x": stagnation_x,   # NEW
                "randomwalk_y": randomwalk_y,   # NEW
                "steps_to_complete": steps_to_complete
            })

    df = pd.DataFrame(detailed_rows)

    # Summary by scenario (include new cols so sheets self-describe settings)
    group_cols = [
        "grid_size", "n_robots", "n_failures", "Number of targets",
        # "pher_deposit", "tau_decay", "pher_min", "bias_alpha",
        "stagnation_x", "randomwalk_y",   # NEW
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

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")
        for sheet_name, frame in [("detailed", df), ("summary", summary)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx-1, idx-1, min(max_len + 2, 60))


if __name__ == "__main__":
    # Example run (keep your original shapes; just pass x,y once)
    for i in range(1,5):
        for j in range(1,5):
    
            run_experiments(
                out_dir="experiments",
                grid_sizes=(25, 50, 75, 100),
                robot_counts=(5, 10, 15, 20),
                failure_counts=(1, 2, 4, 6),
                runs_per_scenario=5,
                n_targets=(80, 100, 120, 150),
                base_seed=7,
                pher_deposit=1.0,
                tau_decay=60.0,     # ignored now
                pher_min=1e-6,
                bias_alpha=1.0,
                uncovered_bonus=10.0,
                stagnation_x=5*i,    # <-- X
                randomwalk_y=5*j,    # <-- Y
            )
