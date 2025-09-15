import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Set
from pathlib import Path

from simulation import FrameWriter, compute_fps, make_writer, run_animation


def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

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
    # print(f' points - {points}')
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

def mark_visible(covered: np.ndarray, x: int, y: int):
    """Mark current cell and its von Neumann neighbors as covered."""
    H, W = covered.shape
    covered[y, x] = True
    if y - 1 >= 0: covered[y - 1, x] = True
    if y + 1 < H:  covered[y + 1, x] = True
    if x - 1 >= 0: covered[y, x - 1] = True
    if x + 1 < W:  covered[y, x + 1] = True

def neighbors_von_neumann(x: int, y: int, W: int, H: int) -> List[Tuple[int,int]]:
    nbs = [(x, y)]
    if y - 1 >= 0: nbs.append((x, y - 1))
    if y + 1 < H:  nbs.append((x, y + 1))
    if x - 1 >= 0: nbs.append((x - 1, y))
    if x + 1 < W:  nbs.append((x + 1, y))
    return nbs

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
    """Create a Manhattan path from start to end."""
    path = [start]
    x1, y1 = start
    x2, y2 = end
    
    # Move horizontally first
    dx = 1 if x2 > x1 else -1
    while x1 != x2:
        x1 += dx
        path.append((x1, y1))
    
    # Then move vertically
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
        if not xs: continue
        # contiguous segments
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

def coverage_to_image(covered_bool: np.ndarray) -> np.ndarray:
    img = np.ones(covered_bool.shape, dtype=float)
    img[covered_bool] = 0.85
    return img

def region_colors(zones: np.ndarray, alpha: float = 0.12) -> Tuple[np.ndarray, dict]:
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab20', np.max(zones)+1)
    rgba = cmap(zones)
    rgba[..., 3] = alpha
    return rgba, {i: cmap(i) for i in np.unique(zones)}

def draw_voronoi_borders(ax: plt.Axes, zones: np.ndarray, color: str = '#0b3d91', lw: float = 1.2, alpha: float = 0.9):
    """
    Draw dark blue borders along edges where adjacent cells belong to different zones.
    """
    H, W = zones.shape
    # Vertical edges between (x,y) and (x+1,y)
    for y in range(H):
        x = 0
        while x < W - 1:
            if zones[y, x] != zones[y, x + 1]:
                # extend a vertical segment if consecutive differing edges align
                x0 = x + 1
                y0 = y
                # vertical segment at x+1 from y..y+1
                ax.plot([x0, x0], [y, y + 1], color=color, linewidth=lw, alpha=alpha, zorder=3)
            x += 1
    # Horizontal edges between (x,y) and (x,y+1)
    for x in range(W):
        y = 0
        while y < H - 1:
            if zones[y, x] != zones[y + 1, x]:
                y0 = y + 1
                ax.plot([x, x + 1], [y0, y0], color=color, linewidth=lw, alpha=alpha, zorder=3)
            y += 1

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Robot:
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0
    failed: bool = False  # <-- NEW

    @property
    def pos(self) -> Tuple[int, int]:
        return self.path[self.idx]

    def step(self):
        # If failed, it stays frozen in place.
        if self.failed:
            return
        if self.idx < len(self.path) - 1:
            self.idx += 1

# -----------------------------
# Simulation step
# -----------------------------
def discover_targets_in_vnhood(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]]):
    """NEW: discover targets in current cell OR its 4-neighborhood."""
    H, W = covered.shape
    for (nx, ny) in neighbors_von_neumann(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))


def _maybe_trigger_failure():
    """Trigger the failure event exactly at FAIL_AT_STEP by freezing the robot."""
    global failure_triggered
    if failure_triggered:
        return
    if FAIL_ROBOT_ID is None or FAIL_AT_STEP is None:
        return
    if global_step == FAIL_AT_STEP:
        r = robots[FAIL_ROBOT_ID]
        r.failed = True
        failure_triggered = True

def _first_finisher_after_failure():
    """
    Return the first (non-failed) robot that has finished its own area (idx at end)
    after the failure happened. If multiple are already finished, the first in list wins.
    """
    if not failure_triggered:
        return None
    for r in robots:
        if r.id == FAIL_ROBOT_ID:
            continue
        if not r.failed and r.idx >= len(r.path) - 1:
            return r
    return None

def _extend_full_pts_for_robot(robot_id: int, extra_path: List[Tuple[int,int]]):
    """
    Update the 'remaining_scatters' full point cache so the animation overlays
    reflect the extended path.
    """
    rem_sc, full_pts = remaining_scatters[robot_id]
    if len(extra_path):
        add_pts = np.array([[x + 0.5, y + 0.5] for (x, y) in extra_path])
        full_pts = np.vstack([full_pts, add_pts])
        remaining_scatters[robot_id] = (rem_sc, full_pts)

def _maybe_reallocate_failed_path():
    """Once a robot finishes after the failure, give it the failed robot's remaining path."""
    global failure_reallocated

    if not failure_triggered or failure_reallocated:
        return

    takeover = _first_finisher_after_failure()
    if takeover is None:
        return

    failed = robots[FAIL_ROBOT_ID]

    # Build extension: from takeover's current (finished) pos to failed's frozen pos,
    # then append the failed robot's remaining path.
    nav = manhattan_path(takeover.pos, failed.pos)          # [start,..., failed.pos]
    rem = failed.path[failed.idx:]                          # [failed.pos, ...]

    # Avoid duplicates at the seam
    extension = nav[1:]  # skip takeover.pos
    if len(rem) > 1:
        extension += rem[1:]  # skip failed.pos (already included by nav)

    # Append to takeover path and update overlays
    takeover.path.extend(extension)
    _extend_full_pts_for_robot(takeover.id, extension)

    # Trim failed robot path to what it actually completed (freeze at current idx)
    failed.path = failed.path[:failed.idx + 1]
    _extend_full_pts_for_robot(failed.id, [])  # keep overlays consistent even if no extra

    failure_reallocated = True

def save_targets_over_time_plot(path: Path):
    """
    Save a dotted line plot: time step (x) vs total targets detected (y, cumulative).
    """
    xs = np.arange(len(targets_found_over_time))
    ys = np.asarray(targets_found_over_time, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, linestyle=':', linewidth=2)  # dotted line
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total targets detected")
    ax.set_title("Targets detected over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sim_step():
    global global_step

    # Check failure BEFORE any robot moves this step, so the failed one won't move.
    _maybe_trigger_failure()

    # Sense, cover, and discover within VN neighborhood at current pose
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y)
        discover_targets_in_vnhood(x, y, targets, found_targets)

    # Move one cell (failed robots won't move)
    for r in robots:
        r.step()

    # Sense, cover, and discover after move (smooth visuals)
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y)
        discover_targets_in_vnhood(x, y, targets, found_targets)

    # After updates for this step, see if any robot finished and can take over
    _maybe_reallocate_failed_path()

    # Tick the global step counter
    global_step += 1

    targets_found_over_time.append(len(found_targets))

def _all_robots_finished():
    return all(r.idx >= len(r.path) - 1 for r in robots)

# -----------------------------
# Animation update
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    img = coverage_to_image(covered)
    world_img.set_data(img)
    shared_img.set_data(img)

    # Robots (positions)
    positions = np.array([[r.pos[0] + 0.5, r.pos[1] + 0.5] for r in robots])
    robot_scatter.set_offsets(positions)

    # NEW: Colors (failed robots red)
    colors = ['red' if r.failed else 'k' for r in robots]
    robot_scatter.set_facecolors(colors)
    robot_scatter.set_edgecolors(colors)

    # NEW: Labels (append " (failed)")
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.pos[0] + 0.6, r.pos[1] + 0.6))
        robot_labels[i].set_text(f"R{r.id}" + (" (failed)" if r.failed else ""))
        robot_labels[i].set_color('red' if r.failed else 'k')

    # Paths overlays
    for i, r in enumerate(robots):
        rem_sc, full_pts = remaining_scatters[i]
        vis_sc, _ = visited_scatters[i]
        v_pts = full_pts[:r.idx+1]
        rem_pts = full_pts[r.idx+1:]
        vis_sc.set_offsets(v_pts if len(v_pts) else np.empty((0, 2)))
        rem_sc.set_offsets(rem_pts if len(rem_pts) else np.empty((0, 2)))

    # Update targets on shared map
    disc = list(found_targets)
    und = list(targets - found_targets)
    if disc:
        dx, dy = zip(*disc)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    else:
        disc_plot.set_offsets(np.empty((0, 2)))
    if und:
        ux, uy = zip(*und)
        und_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
    else:
        und_plot.set_offsets(np.empty((0, 2)))

    ax_world.set_title(
        "World — Equal-Area Voronoi Partition (Centralized)\n"
        f"Covered: {covered.sum()} / {W*H} cells, Found targets: {len(found_targets)} / {len(targets)}"
    )

    global plot_saved
    if (not plot_saved) and _all_robots_finished():
        # pick an output directory; reuse existing if you have one
        try:
            out_dir = OUTPUT_DIR  # if your script already defines it
        except NameError:
            out_dir = Path("out")
        out_dir.mkdir(parents=True, exist_ok=True)

        save_targets_over_time_plot(out_dir / "targets_over_time.png")
        if FAIL_ROBOT_ID is not None:
            np.save('output_metrics/centralized_appraoch_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/centralized_appraoch_without_failure.npy', np.array(targets_found_over_time))
        plot_saved = True

    frame_writer.save(fig)
    return (world_img, shared_img, robot_scatter, und_plot, disc_plot, *robot_labels) + \
           tuple(sc for sc, _ in remaining_scatters) + \
           tuple(sc for sc, _ in visited_scatters)


if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    GRID_SIZE = 25
    N_ROBOTS   = 8
    N_TARGETS  = 30
    STEPS_PER_FRAME = 1
    INTERVAL_MS     = 80
    RANDOM_SEED     = 7

    # for metric - number of targets detected over time steps
    targets_found_over_time = []   # cumulative #found after each sim_step
    plot_saved = False             # guard so we only write once
    OUTPUT_DIR = Path('output_frames/centralized_approach')

    # Failure scenario (NEW)
    FAIL_ROBOT_ID = None#1     # which robot fails
    FAIL_AT_STEP  = None#10  # at which sim step it fails
    global_step = 0
    failure_triggered = False
    failure_reallocated = False

    # Capacitated k-means (balanced power diagram) params
    MAX_ITERS_ASSIGN  = 30
    MAX_ITERS_CENTERS = 10
    LAMBDA_STEP0 = 0.1
    LAMBDA_DECAY = 0.1

    rng = np.random.default_rng(RANDOM_SEED)
    # Pick a frame rate (roughly matches your interactive speed)
    FPS = compute_fps(INTERVAL_MS)
    writer = make_writer(INTERVAL_MS, title="Centralized Swarm", artist="you")
    dir = "output_frames/centralized_approach/"
    frame_writer = FrameWriter(dir)

    # -----------------------------
    # Build scenario
    # -----------------------------
    W = H = GRID_SIZE
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))  # centers

    labels, centers = lloyd_balanced(points, N_ROBOTS, MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN,
                                    LAMBDA_STEP0, LAMBDA_DECAY, rng)
    zones = labels.reshape(H, W)

    # Per-robot masks & sweeping paths
    masks = [(zones == i) for i in range(N_ROBOTS)]
    sweeping_paths: List[List[Tuple[int,int]]] = []
    for i in range(N_ROBOTS):
        p = sensor_aware_path_for_region(masks[i])
        if not p:
            cx, cy = centers[i]
            x = int(np.clip(round(cx - 0.5), 0, W-1))
            y = int(np.clip(round(cy - 0.5), 0, H-1))
            p = [(x, y)]
        sweeping_paths.append(p)

    # Create full paths: center -> first cell of sweeping path -> sweeping path
    center_pos = (W // 2, H // 2)
    full_paths: List[List[Tuple[int,int]]] = []
    
    for i in range(N_ROBOTS):
        sweeping_path = sweeping_paths[i]
        if sweeping_path:
            # Create path from center to first position of sweeping path
            nav_path = manhattan_path(center_pos, sweeping_path[0])
            # Combine navigation path (excluding the duplicate end point) with sweeping path
            full_path = nav_path[:-1] + sweeping_path
        else:
            # Fallback if no sweeping path
            full_path = [center_pos]
        
        full_paths.append(full_path)

    robots = [Robot(i, full_paths[i]) for i in range(N_ROBOTS)]

    covered = np.zeros((H, W), dtype=bool)
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    found_targets: Set[Tuple[int, int]] = set()

    # -----------------------------
    # Matplotlib setup
    # -----------------------------

    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_shared = fig.add_subplot(gs[0, 1])

    # Faint Voronoi regions + coverage
    zone_rgba, _ = region_colors(zones, alpha=0.12)
    ax_world.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
    world_img = ax_world.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0.0, vmax=1.0)

    # Draw dark blue Voronoi borders (NEW)
    draw_voronoi_borders(ax_world, zones, color='#003366', lw=1.2, alpha=0.9)

    ax_world.set_title("World — Equal-Area Voronoi Partition (Centralized)")
    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')
    ax_world.set_xticks(np.arange(0, W+1, 10)); ax_world.set_yticks(np.arange(0, H+1, 10))
    ax_world.set_xticks(np.arange(0, W+1, 1), minor=True); ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    # Robots (colored markers)
    robot_colors = ['k' for _ in robots]  # black initially
    robot_scatter = ax_world.scatter([r.pos[0] + 0.5 for r in robots],
                                    [r.pos[1] + 0.5 for r in robots],
                                    s=40, marker='o', c=robot_colors, zorder=4)

    # Labels above robots
    robot_labels = []
    for r in robots:
        txt = ax_world.text(r.pos[0] + 0.6, r.pos[1] + 0.6,
                            f"R{r.id}", fontsize=7, color='k', zorder=6)
        robot_labels.append(txt)

    # Paths (remaining vs visited)
    remaining_scatters, visited_scatters = [], []
    for r in robots:
        full_pts = np.array([[x + 0.5, y + 0.5] for (x, y) in r.path])
        rem_sc = ax_world.scatter(full_pts[:, 0], full_pts[:, 1], s=6, marker='s',
                                facecolors='none', edgecolors='0.35', alpha=0.14, linewidths=0.45, zorder=1)
        vis_sc = ax_world.scatter([], [], s=10, marker='s',
                                facecolors='none', edgecolors='0.2', alpha=0.55, linewidths=0.6, zorder=2)
        remaining_scatters.append((rem_sc, full_pts))
        visited_scatters.append((vis_sc, []))

    # Targets in world
    if targets:
        tx_world, ty_world = zip(*targets)
    else:
        tx_world, ty_world = [], []
    ax_world.scatter([x + 0.5 for x in tx_world], [y + 0.5 for y in ty_world],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)

    # Shared map (same overlays)
    ax_shared.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
    shared_img = ax_shared.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0.0, vmax=1.0)
    # draw_voronoi_borders(ax_shared, zones, color='#003366', lw=1.2, alpha=0.9)

    ax_shared.set_title("Shared Map (Global Knowledge)")
    ax_shared.set_xlim(0, W); ax_shared.set_ylim(0, H); ax_shared.set_aspect('equal', adjustable='box')
    ax_shared.set_xticks(np.arange(0, W+1, 10)); ax_shared.set_yticks(np.arange(0, H+1, 10))
    ax_shared.set_xticks(np.arange(0, W+1, 1), minor=True); ax_shared.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_shared.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_shared.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    disc_plot = ax_shared.scatter([], [], s=25, marker='o',
                                facecolors='none', edgecolors='g', linewidths=1.5, label='Discovered')
    und_tx, und_ty = (list(zip(*targets)) if targets else ([], []))
    und_plot = ax_shared.scatter([x + 0.5 for x in und_tx], [y + 0.5 for y in und_ty],
                                s=18, marker='x', c='r', label='Undiscovered')
    ax_shared.legend(loc='upper right', fontsize=8, frameon=False)


    # -----------------------------
    # Run
    # -----------------------------

    anim = run_animation(fig, update, frames=2000000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()