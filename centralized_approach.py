import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Set

# -----------------------------
# Parameters
# -----------------------------
GRID_SIZE = 50
N_ROBOTS   = 7
N_TARGETS  = 20
STEPS_PER_FRAME = 1
INTERVAL_MS     = 80
RANDOM_SEED     = 7

# Capacitated k-means (balanced power diagram) params
MAX_ITERS_ASSIGN = 30      # assignment / lambda updates
MAX_ITERS_CENTERS = 10     # center updates (outer Lloyd steps)
LAMBDA_STEP0 = 0.1         # initial step-size for lambda updates
LAMBDA_DECAY = 0.9         # decay per iteration

rng = np.random.default_rng(RANDOM_SEED)

# -----------------------------
# Utility
# -----------------------------
def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

def kpp_init(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initializer for centers (on continuous coordinates)."""
    n = points.shape[0]
    centers = np.empty((k, 2), dtype=float)
    # pick first
    idx = rng.integers(n)
    centers[0] = points[idx]
    # pick remaining
    d2 = np.full(n, np.inf)
    for i in range(1, k):
        # update distances to nearest chosen center
        d2 = np.minimum(d2, np.sum((points - centers[i-1])**2, axis=1))
        # choose new with prob proportional to d2
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
    """
    Assign each point to argmin_k ||x - c_k||^2 + lambda_k, then update lambdas to push
    cluster sizes toward 'target'. Returns (labels, lambdas).
    """
    k = centers.shape[0]
    lambdas = np.zeros(k, dtype=float)
    labels = np.zeros(points.shape[0], dtype=int)

    step = step0
    for t in range(iters):
        # Assignment
        # cost[i,k] = ||x_i - c_k||^2 + lambda_k
        diffs = points[:, None, :] - centers[None, :, :]    # (N,K,2)
        d2 = np.sum(diffs**2, axis=2)                      # (N,K)
        costs = d2 + lambdas[None, :]
        labels = np.argmin(costs, axis=1)

        # Subgradient update for lambdas: lambda_k += step * (size_k - target)
        sizes = np.bincount(labels, minlength=k).astype(float)
        lambdas += step * (sizes - target)

        # mild damping
        step *= decay

    return labels, lambdas

def lloyd_balanced(points: np.ndarray, k: int, grid_W: int, grid_H: int,
                   max_iters_centers: int, max_iters_assign: int,
                   step0: float, decay: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced Lloyd: alternates between balanced assignment and center recomputation.
    """
    centers = kpp_init(points, k, rng)

    target = (points.shape[0] // k)  # equal area target per region (rough)
    for _ in range(max_iters_centers):
        labels, lambdas = balanced_power_diagram_assign(
            points, centers, target=target,
            iters=max_iters_assign, step0=step0, decay=decay, rng=rng
        )
        # Update centers to mean of assigned points (keep inside grid)
        new_centers = np.empty_like(centers)
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) > 0:
                new_centers[j] = points[idx].mean(axis=0)
            else:
                # re-seed a lonely center to a random point
                new_centers[j] = points[rng.integers(points.shape[0])]
        # stop if stable
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    # Final assignment with last centers
    labels, lambdas = balanced_power_diagram_assign(
        points, centers, target=target,
        iters=max_iters_assign, step0=step0, decay=decay, rng=rng
    )
    return labels, centers

def mark_visible(covered: np.ndarray, x: int, y: int):
    """Mark current cell and its von Neumann neighbors as covered."""
    H, W = covered.shape
    covered[y, x] = True
    if y - 1 >= 0: covered[y - 1, x] = True
    if y + 1 < H:  covered[y + 1, x] = True
    if x - 1 >= 0: covered[y, x - 1] = True
    if x + 1 < W:  covered[y, x + 1] = True

def manhattan_connect(path: List[Tuple[int,int]], x2: int, y2: int):
    """Append step-by-step Manhattan moves from last point in path to (x2,y2)."""
    if not path:
        return
    x1, y1 = path[-1]
    # Horizontal first, then vertical (choice is arbitrary; keeps 1-cell steps)
    dx = 1 if x2 > x1 else -1
    while x1 != x2:
        x1 += dx
        path.append((x1, y1))
    dy = 1 if y2 > y1 else -1
    while y1 != y2:
        y1 += dy
        path.append((x1, y1))

def sensor_aware_path_for_region(mask: np.ndarray) -> List[Tuple[int,int]]:
    """
    Build a continuous 1-cell-per-step path restricted to 'mask' cells:
    - visit rows y = 0,2,4,...;
    - within a visited row, traverse the contiguous x-interval(s) belonging to the region
      left->right then right->left alternating;
    - connect between segments and rows via Manhattan moves.
    """
    H, W = mask.shape
    path: List[Tuple[int,int]] = []
    lr = True  # direction flip flag per visited row
    started = False
    for y in range(0, H, 2):
        xs = [x for x in range(W) if mask[y, x]]
        if not xs:
            continue
        # Break into contiguous segments for safety (usually one)
        segments = []
        start = xs[0]
        prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                segments.append((start, prev))
                start = x
                prev = x
        segments.append((start, prev))

        # Traverse segments in chosen direction
        segs = segments if lr else list(reversed(segments))
        for (a, b) in segs:
            run = list(range(a, b + 1)) if lr else list(range(b, a - 1, -1))
            # Move to segment start if needed
            if not started:
                path.append((run[0], y))
                started = True
            else:
                manhattan_connect(path, run[0], y)
            # Sweep segment
            for x in run[1:]:
                path.append((x, y))
        lr = not lr

        # Connect to next visited row y+2 if it exists and has cells; choose nearest x
        if y + 2 < H:
            xs_next = [x for x in range(W) if mask[y + 2, x]]
            if xs_next:
                # choose the nearest x in next row to current endpoint
                x_curr, y_curr = path[-1]
                x_next = min(xs_next, key=lambda xx: abs(xx - x_curr))
                manhattan_connect(path, x_next, y + 2)

    # Ensure uniqueness of consecutive positions (already ensured by construction)
    return path

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Robot:
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0  # current path index

    @property
    def pos(self) -> Tuple[int, int]:
        return self.path[self.idx]

    def step(self):
        if self.idx < len(self.path) - 1:
            self.idx += 1

# -----------------------------
# Build scenario
# -----------------------------
W = H = GRID_SIZE

# All cell centers (continuous coords)
yy, xx = np.mgrid[0:H, 0:W]
points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))  # (N,2), N=10000

# Balanced Voronoi partition via balanced Lloyd iterations
labels, centers = lloyd_balanced(points, N_ROBOTS, W, H,
                                 MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN,
                                 LAMBDA_STEP0, LAMBDA_DECAY, rng)

zones = labels.reshape(H, W)   # region id per cell [0..N_ROBOTS-1]

# Build per-robot masks and paths
masks = [(zones == i) for i in range(N_ROBOTS)]
paths: List[List[Tuple[int,int]]] = []
for i in range(N_ROBOTS):
    p = sensor_aware_path_for_region(masks[i])
    # Safety: if a region is empty (rare), skip; else ensure path isn't empty
    if not p:
        # Fallback: pick the nearest grid cell to the center
        cx, cy = centers[i]
        x, y = int(np.clip(round(cx - 0.5), 0, W-1)), int(np.clip(round(cy - 0.5), 0, H-1))
        p = [(x, y)]
    paths.append(p)

robots = [Robot(i, paths[i]) for i in range(N_ROBOTS)]

# State
covered = np.zeros((H, W), dtype=bool)
targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
found_targets: Set[Tuple[int, int]] = set()

# -----------------------------
# Visualization helpers
# -----------------------------
def coverage_to_image(covered_bool: np.ndarray) -> np.ndarray:
    img = np.ones(covered_bool.shape, dtype=float)
    img[covered_bool] = 0.85  # light gray
    return img

def region_colors(zones: np.ndarray, alpha: float = 0.12) -> Tuple[np.ndarray, dict]:
    """Map zones -> RGBA colors with faint alpha overlay."""
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab20', N_ROBOTS)
    rgba = cmap(zones % N_ROBOTS)
    rgba[..., 3] = alpha
    return rgba, {i: cmap(i) for i in range(N_ROBOTS)}

# -----------------------------
# Matplotlib setup
# -----------------------------
fig = plt.figure(figsize=(12.0, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
ax_world = fig.add_subplot(gs[0, 0])
ax_shared = fig.add_subplot(gs[0, 1])

# Bottom layer: faint Voronoi colors
zone_rgba, zone_cmap = region_colors(zones, alpha=0.12)
zone_img_world = ax_world.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
# Coverage layer on top
world_img = ax_world.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0.0, vmax=1.0)

ax_world.set_title("World — Equal-Area Voronoi Partition (Centralized)")
ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')

# Faint grid
ax_world.set_xticks(np.arange(0, W+1, 10))
ax_world.set_yticks(np.arange(0, H+1, 10))
ax_world.set_xticks(np.arange(0, W+1, 1), minor=True)
ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

# Robots (dots)
robot_scatter = ax_world.scatter(
    [r.pos[0] + 0.5 for r in robots],
    [r.pos[1] + 0.5 for r in robots],
    s=40, marker='o', c='k', zorder=3
)

# Per-robot paths (remaining vs visited)
remaining_scatters = []
visited_scatters = []
for r in robots:
    full_pts = np.array([[x + 0.5, y + 0.5] for (x, y) in r.path])
    rem_sc = ax_world.scatter(
        full_pts[:, 0], full_pts[:, 1], s=6, marker='s',
        facecolors='none', edgecolors='0.35', alpha=0.14, linewidths=0.45, zorder=1
    )
    vis_sc = ax_world.scatter(
        [], [], s=10, marker='s',
        facecolors='none', edgecolors='0.2', alpha=0.55, linewidths=0.6, zorder=2
    )
    remaining_scatters.append((rem_sc, full_pts))
    visited_scatters.append((vis_sc, []))

# Targets in world
if targets:
    tx_world, ty_world = zip(*targets)
else:
    tx_world, ty_world = [], []
targets_world_plot = ax_world.scatter(
    [x + 0.5 for x in tx_world], [y + 0.5 for y in ty_world],
    s=20, marker='x', c='r', alpha=0.9, zorder=4
)

# Shared map (show same faint zones + coverage + target state)
zone_img_shared = ax_shared.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
shared_img = ax_shared.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0.0, vmax=1.0)

ax_shared.set_title("Shared Map (Global Knowledge)")
ax_shared.set_xlim(0, W); ax_shared.set_ylim(0, H); ax_shared.set_aspect('equal', adjustable='box')
ax_shared.set_xticks(np.arange(0, W+1, 10))
ax_shared.set_yticks(np.arange(0, H+1, 10))
ax_shared.set_xticks(np.arange(0, W+1, 1), minor=True)
ax_shared.set_yticks(np.arange(0, H+1, 1), minor=True)
ax_shared.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
ax_shared.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

disc_plot = ax_shared.scatter([], [], s=25, marker='o',
                              facecolors='none', edgecolors='g', linewidths=1.5, label='Discovered')
und_tx, und_ty = (list(zip(*targets)) if targets else ([], []))
und_plot = ax_shared.scatter([x + 0.5 for x in und_tx], [y + 0.5 for y in und_ty],
                             s=18, marker='x', c='r', label='Undiscovered')
ax_shared.legend(loc='upper right', fontsize=8, frameon=False)

# -----------------------------
# Simulation step
# -----------------------------
def sim_step():
    # Sense & cover at current pose
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y)
        if (x, y) in targets:
            found_targets.add((x, y))

    # Move one cell
    for r in robots:
        r.step()

    # Sense & cover after move (smooth visuals)
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y)
        if (x, y) in targets:
            found_targets.add((x, y))

# -----------------------------
# Animation update
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    img = coverage_to_image(covered)
    world_img.set_data(img)
    shared_img.set_data(img)

    # Robots
    robot_scatter.set_offsets(np.array([[r.pos[0] + 0.5, r.pos[1] + 0.5] for r in robots]))

    # Paths
    for i, r in enumerate(robots):
        rem_sc, full_pts = remaining_scatters[i]
        vis_sc, _ = visited_scatters[i]
        v_pts = full_pts[:r.idx+1]
        rem_pts = full_pts[r.idx+1:]
        vis_sc.set_offsets(v_pts if len(v_pts) else np.empty((0, 2)))
        rem_sc.set_offsets(rem_pts if len(rem_pts) else np.empty((0, 2)))

    # Targets on shared
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
    return (world_img, shared_img, robot_scatter, und_plot, disc_plot) + \
           tuple(sc for sc, _ in remaining_scatters) + \
           tuple(sc for sc, _ in visited_scatters)

# -----------------------------
# Run
# -----------------------------
anim = FuncAnimation(fig, update, frames=200000, interval=INTERVAL_MS, blit=False)
plt.show()
