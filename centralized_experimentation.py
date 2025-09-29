# centralized_experimentation.py
# Headless experimentation for centralized approach with multi-robot failures (FIFO takeovers)
# Produces an Excel file with results aligned to stigmergy_experimentation.py experiments.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Dict, Iterable, Optional
import numpy as np
import pandas as pd

# -----------------------------
# Core centralized utilities
# -----------------------------

def kpp_init(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    centers = np.empty((k, 2), dtype=float)
    idx = rng.integers(n)
    centers[0] = points[idx]
    d2 = np.full(n, np.inf)
    for i in range(1, k):
        d2 = np.minimum(d2, np.sum((points - centers[i-1])**2, axis=1))
        probs = d2 / d2.sum()
        idx = rng.choice(n, p=probs)
        centers[i] = points[idx]
    return centers

def balanced_power_diagram_assign(points: np.ndarray,
                                  centers: np.ndarray,
                                  target: int,
                                  iters: int,
                                  step0: float,
                                  decay: float,
                                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    k = centers.shape[0]
    lambdas = np.zeros(k, dtype=float)
    labels = np.zeros(points.shape[0], dtype=int)
    step = step0
    for _ in range(iters):
        diffs = points[:, None, :] - centers[None, :, :]
        d2 = np.sum(diffs**2, axis=2)
        costs = d2 + lambdas[None, :]
        labels = np.argmin(costs, axis=1)
        sizes = np.bincount(labels, minlength=k).astype(float)
        lambdas += step * (sizes - target)
        step *= decay
    return labels, lambdas

def lloyd_balanced(points: np.ndarray, k: int,
                   max_iters_centers: int, max_iters_assign: int,
                   step0: float, decay: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    centers = kpp_init(points, k, rng)
    target = (points.shape[0] // k)
    for _ in range(max_iters_centers):
        labels, _ = balanced_power_diagram_assign(points, centers, target, max_iters_assign, step0, decay, rng)
        new_centers = np.empty_like(centers)
        for j in range(k):
            idx = np.where(labels == j)[0]
            new_centers[j] = points[idx].mean(axis=0) if len(idx) > 0 else points[rng.integers(points.shape[0])]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    labels, _ = balanced_power_diagram_assign(points, centers, target, max_iters_assign, step0, decay, rng)
    return labels, centers

def manhattan_connect(path: List[Tuple[int,int]], x2: int, y2: int):
    if not path:
        return
    x1, y1 = path[-1]
    dx = 1 if x2 > x1 else -1
    while x1 != x2:
        x1 += dx
        path.append((x1, y1))
    dy = 1 if y2 > y1 else -1
    while y1 != y2:
        y1 += dy
        path.append((x1, y1))

def manhattan_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [start]
    x1, y1 = start
    x2, y2 = end
    dx = 1 if x2 > x1 else -1
    while x1 != x2:
        x1 += dx
        path.append((x1, y1))
    dy = 1 if y2 > y1 else -1
    while y1 != y2:
        y1 += dy
        path.append((x1, y1))
    return path

def sensor_aware_path_for_region(mask: np.ndarray) -> List[Tuple[int,int]]:
    H, W = mask.shape
    path: List[Tuple[int,int]] = []
    lr = True
    started = False
    for y in range(0, H, 2):
        xs = [x for x in range(W) if mask[y, x]]
        if not xs: 
            continue
        # contiguous segments in row y
        segs = []
        s = xs[0]; p = xs[0]
        for x in xs[1:]:
            if x == p + 1:
                p = x
            else:
                segs.append((s, p)); s = x; p = x
        segs.append((s, p))
        segs = segs if lr else list(reversed(segs))
        for (a, b) in segs:
            run = list(range(a, b + 1)) if lr else list(range(b, a - 1, -1))
            if not started:
                path.append((run[0], y)); started = True
            else:
                manhattan_connect(path, run[0], y)
            for x in run[1:]:
                path.append((x, y))
        lr = not lr
        if y + 2 < H:
            xs_next = [x for x in range(W) if mask[y + 2, x]]
            if xs_next:
                x_curr, _ = path[-1]
                x_next = min(xs_next, key=lambda xx: abs(xx - x_curr))
                manhattan_connect(path, x_next, y + 2)
    return path

def neighbors_von_neumann(x: int, y: int, W: int, H: int) -> List[Tuple[int,int]]:
    nbs = [(x, y)]
    if y - 1 >= 0: nbs.append((x, y - 1))
    if y + 1 < H:  nbs.append((x, y + 1))
    if x - 1 >= 0: nbs.append((x - 1, y))
    if x + 1 < W:  nbs.append((x + 1, y))
    return nbs

def mark_visible(covered: np.ndarray, x: int, y: int):
    H, W = covered.shape
    covered[y, x] = True
    if y - 1 >= 0: covered[y - 1, x] = True
    if y + 1 < H:  covered[y + 1, x] = True
    if x - 1 >= 0: covered[y, x - 1] = True
    if x + 1 < W:  covered[y, x + 1] = True

def discover_targets_in_vnhood(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_von_neumann(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

def generate_unique_targets(grid_size: int, m: int, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

@dataclass
class Robot:
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0
    failed: bool = False

    @property
    def pos(self) -> Tuple[int, int]:
        return self.path[self.idx]

    def step(self):
        if self.failed:
            return
        if self.idx < len(self.path) - 1:
            self.idx += 1

# -----------------------------
# Headless centralized simulation (no plotting)
# -----------------------------

def _takeover_append_path(robots: List[Robot], takeover_id: int, failed_id: int):
    takeover = robots[takeover_id]
    failed = robots[failed_id]

    # nav to failed.pos + append failed remaining (skip seam duplicates)
    nav = manhattan_path(takeover.pos, failed.pos)   # includes takeover.pos and failed.pos
    rem = failed.path[failed.idx:]                   # includes failed.pos

    extension: List[Tuple[int,int]] = []
    if len(nav) > 1:
        extension += nav[1:]                         # skip duplicate takeover.pos
    if len(rem) > 1:
        extension += rem[1:]                         # skip duplicate failed.pos

    if extension:
        takeover.path.extend(extension)

    # Trim failed's path to reached index
    failed.path = failed.path[:failed.idx + 1]

def run_simulation(
    grid_size: int,
    n_robots: int,
    n_targets: int,
    failure_schedule: List[Tuple[int, int]],
    rng_seed: int,
    max_steps: Optional[int] = None,
    # Lloyd/balanced partitions
    MAX_ITERS_ASSIGN: int = 30,
    MAX_ITERS_CENTERS: int = 10,
    LAMBDA_STEP0: float = 0.1,
    LAMBDA_DECAY: float = 0.1,
) -> Dict[str, object]:
    """
    Returns:
      steps_to_complete: int
      failure_schedule: List[Tuple[robot_id, step]] (normalized, sorted)
    """
    rng = np.random.default_rng(rng_seed)
    W = H = grid_size
    if max_steps is None:
        max_steps = int(W * H * 10)

    # Build grid points and balanced zones (centralized logic)
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    labels, centers = lloyd_balanced(points, n_robots, MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN,
                                     LAMBDA_STEP0, LAMBDA_DECAY, rng)
    zones = labels.reshape(H, W)

    # Per-robot sweeping paths
    masks = [(zones == i) for i in range(n_robots)]
    sweeping_paths: List[List[Tuple[int,int]]] = []
    for i in range(n_robots):
        p = sensor_aware_path_for_region(masks[i])
        if not p:
            cx, cy = centers[i]
            x = int(np.clip(round(cx - 0.5), 0, W-1))
            y = int(np.clip(round(cy - 0.5), 0, H-1))
            p = [(x, y)]
        sweeping_paths.append(p)

    # Full paths: center -> first sweep cell -> sweep
    center_pos = (W // 2, H // 2)
    full_paths: List[List[Tuple[int,int]]] = []
    for i in range(n_robots):
        sp = sweeping_paths[i]
        if sp:
            nav = manhattan_path(center_pos, sp[0])
            full_paths.append(nav[:-1] + sp)
        else:
            full_paths.append([center_pos])

    robots = [Robot(i, full_paths[i]) for i in range(n_robots)]

    # Targets & coverage
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Tuple[int,int]] = set()
    covered = np.zeros((H, W), dtype=bool)

    # Normalize and map schedule: step -> [rids] (sorted deterministic)
    norm_sched: List[Tuple[int,int]] = []
    for rid, st in (failure_schedule or []):
        if 0 <= rid < n_robots and st is not None and st >= 0:
            norm_sched.append((int(rid), int(st)))
    norm_sched.sort(key=lambda x: (x[1], x[0]))
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)

    # FIFO queue of pending failed robots; assign when a finisher exists
    from collections import deque
    pending_failures = deque()

    steps = 0
    while steps < max_steps:
        # Trigger failures for this step BEFORE any move
        if steps in fail_map:
            for rid in fail_map[steps]:
                r = robots[rid]
                if not r.failed:   # freeze in place
                    r.failed = True
                    pending_failures.append(rid)

        # Sense/cover/discover at current pose
        for r in robots:
            x, y = r.pos
            mark_visible(covered, x, y)
            discover_targets_in_vnhood(x, y, targets, found_targets, W, H)

        # Move one cell
        for r in robots:
            r.step()

        # Sense/cover/discover after move (smooth parity with plotted version)
        for r in robots:
            x, y = r.pos
            mark_visible(covered, x, y)
            discover_targets_in_vnhood(x, y, targets, found_targets, W, H)

        # Assign finishers to failed robots (FIFO), one failed per finisher
        if pending_failures:
            finishers = [rr.id for rr in robots if (not rr.failed) and (rr.idx >= len(rr.path) - 1)]
            for fin_id in finishers:
                if not pending_failures:
                    break
                failed_id = pending_failures.popleft()
                _takeover_append_path(robots, fin_id, failed_id)

        steps += 1

        # Stop as soon as ALL targets discovered
        if len(found_targets) >= len(targets):
            break

    return {
        "steps_to_complete": int(steps),
        "failure_schedule": norm_sched,
        "n_targets": int(len(targets)),
    }

# -----------------------------
# Failure schedule (match stigmergy_experimentation.py)
# -----------------------------

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

# -----------------------------
# Experiment driver (Excel output)
# -----------------------------

def run_experiments(
    out_dir: str = "experiments",
    grid_sizes: Iterable[int] = (25, 50, 75, 100),
    robot_counts: Iterable[int] = (5, 10, 15, 20),
    failure_counts: Iterable[int] = (1, 2, 4, 6),
    runs_per_scenario: int = 5,
    n_targets: Iterable[int] = (20, 30, 40, 50),
    base_seed: int = 7,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_path / "experiments_centralized.xlsx"

    rows: List[Dict[str, object]] = []

    # For apples-to-apples, mirror the stigmergy schedule seeding pattern:
    # schedule_seed = (base_seed * 10_000) + (G * 100) + (R * 10) + (F * 1000)
    for i in range(len(grid_sizes)):
        G = int(grid_sizes[i])
        R = int(robot_counts[i])
        F = int(failure_counts[i])
        T = int(n_targets[i])

        schedule_seed = (base_seed * 10_000) + (G * 100) + (R * 10) + (F * 1000)
        rng_sched = np.random.default_rng(schedule_seed)
        max_steps_hint = max(G * G // max(1, R), 200)
        failure_schedule = make_random_failure_schedule(R, F, rng_sched, max_steps_hint)

        for run_idx in range(1, runs_per_scenario + 1):
            run_seed = schedule_seed #+ run_idx  # same pattern used in stigmergy experiments

            result = run_simulation(
                grid_size=G,
                n_robots=R,
                n_targets=T,
                failure_schedule=failure_schedule,
                rng_seed=run_seed,
                max_steps=int(G * G * 10)
            )

            steps_to_complete = result["steps_to_complete"]
            fail_ids   = [rid for rid, _ in result["failure_schedule"]]
            fail_steps = [st  for _,  st in result["failure_schedule"]]
            rows.append({
                "grid_size": G,
                "n_robots": R,
                "n_failures": F,
                # "run_idx": run_idx,
                # "seed": run_seed,
                "targets": result["n_targets"],
                "failed_robot_ids": ";".join(map(str, fail_ids)),
                "fail_steps": ";".join(map(str, fail_steps)),
                "steps_to_complete": steps_to_complete,
            })

    df = pd.DataFrame(rows)

    # Optional summary
    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures"], as_index=False)
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

        # Auto-fit columns
        for sheet_name, frame in [("detailed", df), ("summary", summary)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx-1, idx-1, min(max_len + 2, 60))

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    run_experiments(
        out_dir="experiments",
        grid_sizes=(25, 50, 75, 100),
        robot_counts=(5, 10, 15, 20),
        failure_counts=(1, 2, 4, 6),
        runs_per_scenario=1,
        n_targets=(80, 100, 120, 150),
        base_seed=7,
    )
