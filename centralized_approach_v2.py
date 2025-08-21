
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional


# =========================
# Utilities
# =========================
def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[int(i)] for i in choices)

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
    target = points.shape[0] // k
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


def concat_paths(a: List[Tuple[int,int]], b: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not a: return b.copy()
    if not b: return a.copy()
    out = a.copy()
    ax, ay = out[-1]
    bx, by = b[0]
    if (ax, ay) != (bx, by):
        out += manhattan_path(ax, ay, bx, by)
    else:
        b = b[1:]
    out += b
    return out

def neighbors_vn(x: int, y: int, W: int, H: int) -> List[Tuple[int,int]]:
    out = [(x, y)]
    if y-1 >= 0: out.append((x, y-1))
    if y+1 < H:  out.append((x, y+1))
    if x-1 >= 0: out.append((x-1, y))
    if x+1 < W:  out.append((x+1, y))
    return out

def mark_visible(covered: np.ndarray, x: int, y: int):
    H, W = covered.shape
    covered[y, x] = True
    if y-1 >= 0: covered[y-1, x] = True
    if y+1 < H:  covered[y+1, x] = True
    if x-1 >= 0: covered[y, x-1] = True
    if x+1 < W:  covered[y, x+1] = True

def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_vn(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

def manhattan_path(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int,int]]:
    path: List[Tuple[int,int]] = []
    x, y = x1, y1
    if x2 != x:
        step = 1 if x2 > x else -1
        while x != x2:
            x += step
            path.append((x, y))
    if y2 != y:
        step = 1 if y2 > y else -1
        while y != y2:
            y += step
            path.append((x, y))
    return path

def get_region_bbox(mask: np.ndarray) -> Tuple[int,int,int,int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

def corner_candidates(min_x, max_x, min_y, max_y):
    return {
        "BL": (min_x, min_y),
        "TL": (min_x, max_y),
        "BR": (max_x, min_y),
        "TR": (max_x, max_y),
    }

def find_anchor_near_point(mask: np.ndarray, cx: int, cy: int) -> Tuple[int,int]:
    ys, xs = np.where(mask)
    if len(xs) == 0: raise ValueError("Empty region mask.")
    d = np.abs(xs - cx) + np.abs(ys - cy)
    k = int(np.argmin(d))
    return int(xs[k]), int(ys[k])

def rotate_path_to_exact_start(path: List[Tuple[int,int]], start_cell: Tuple[int,int]) -> List[Tuple[int,int]]:
    for i, p in enumerate(path):
        if p == start_cell:
            return path[i:] + path[:i]
    return []

def find_adjacent_path(mask: np.ndarray, start: Tuple[int,int], end: Tuple[int,int]) -> List[Tuple[int,int]]:
    """Find a path between start and end using only adjacent moves within the masked region.
    Uses BFS to find shortest path of 4-connected moves."""
    if start == end:
        return [start]
    
    H, W = mask.shape
    sx, sy = start
    ex, ey = end
    
    # BFS to find path
    from collections import deque
    queue = deque([(sx, sy, [(sx, sy)])])
    visited = {(sx, sy)}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
    
    while queue:
        x, y, path = queue.popleft()
        
        if (x, y) == (ex, ey):
            return path
            
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds and if cell is in region and not visited
            if (0 <= nx < W and 0 <= ny < H and 
                mask[ny, nx] and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny, path + [(nx, ny)]))
    
    # If no path found within region, return None
    return None
    if not a: return b.copy()
    if not b: return a.copy()
    out = a.copy()
    ax, ay = out[-1]
    bx, by = b[0]
    if (ax, ay) != (bx, by):
        out += manhattan_path(ax, ay, bx, by)
    else:
        b = b[1:]
    out += b
    return out

# -------------------------
# Fixed stripe helpers (continuous)
# -------------------------
def contiguous_segments_1d(indices: List[int]) -> List[Tuple[int,int]]:
    if not indices: return []
    segs = []
    s = indices[0]; p = indices[0]
    for v in indices[1:]:
        if v == p + 1:
            p = v
        else:
            segs.append((s, p)); s = v; p = v
    segs.append((s, p))
    return segs

def sweep_columns(mask: np.ndarray, xs: List[int], start_dir_up: bool) -> List[Tuple[int,int]]:
    """Column stripes; traverse each column up/down, connect between columns only to adjacent cells."""
    H, W = mask.shape
    path: List[Tuple[int,int]] = []
    dir_up = start_dir_up
    last_pos = None
    
    for x in xs:
        ys = [y for y in range(H) if mask[y, x]]
        if not ys:
            continue
            
        segs = contiguous_segments_1d(ys)
        if not dir_up:
            segs = list(reversed(segs))
        
        column_start = None
        column_end = None
        
        for seg_idx, (a, b) in enumerate(segs):
            run = (list(range(a, b + 1)) if dir_up else list(range(b, a - 1, -1)))
            
            if seg_idx == 0:
                column_start = (x, run[0])
                # Connect from last position if this isn't the first column
                if last_pos is not None:
                    # Only connect if adjacent (distance of 1 in x or y direction)
                    dist = abs(last_pos[0] - column_start[0]) + abs(last_pos[1] - column_start[1])
                    if dist == 1:
                        # Adjacent - can move directly
                        if path and path[-1] != column_start:
                            path.append(column_start)
                    else:
                        # Not adjacent - need to find path along region boundary
                        bridge = find_adjacent_path(mask, last_pos, column_start)
                        if bridge:
                            # Skip first element if it's already in path
                            start_idx = 1 if path and path[-1] == bridge[0] else 0
                            path.extend(bridge[start_idx:])
                        else:
                            # Fallback to manhattan if no adjacent path found
                            path += manhattan_path(last_pos[0], last_pos[1], column_start[0], column_start[1])
                else:
                    # First column
                    path.append(column_start)
                
                # Add rest of first segment
                for y in run[1:]:
                    path.append((x, y))
            else:
                # Connect segments within the same column
                prev_end = path[-1]
                seg_start = (x, run[0])
                # Find path between segments in same column
                bridge = find_adjacent_path(mask, prev_end, seg_start)
                if bridge:
                    # Skip first element since it's the previous end
                    path.extend(bridge[1:])
                else:
                    # Fallback to manhattan
                    path += manhattan_path(prev_end[0], prev_end[1], seg_start[0], seg_start[1])
                
                # Add rest of segment (skip first since it's already added)
                for y in run[1:]:
                    path.append((x, y))
            
            column_end = (x, run[-1])
        
        if column_end is not None:
            last_pos = column_end
        dir_up = not dir_up
    return path

def sweep_rows(mask: np.ndarray, ys: List[int], start_dir_lr: bool) -> List[Tuple[int,int]]:
    """Row stripes; traverse each row left/right, connect between rows only to adjacent cells."""
    H, W = mask.shape
    path: List[Tuple[int,int]] = []
    dir_lr = start_dir_lr
    last_pos = None
    
    for y in ys:
        xs = [x for x in range(W) if mask[y, x]]
        if not xs:
            continue
            
        segs = contiguous_segments_1d(xs)
        if not dir_lr:
            segs = list(reversed(segs))
        
        row_start = None
        row_end = None
        
        for seg_idx, (a, b) in enumerate(segs):
            run = (list(range(a, b + 1)) if dir_lr else list(range(b, a - 1, -1)))
            
            if seg_idx == 0:
                row_start = (run[0], y)
                # Connect from last position if this isn't the first row
                if last_pos is not None:
                    # Only connect if adjacent (distance of 1 in x or y direction)
                    dist = abs(last_pos[0] - row_start[0]) + abs(last_pos[1] - row_start[1])
                    if dist == 1:
                        # Adjacent - can move directly
                        if path and path[-1] != row_start:
                            path.append(row_start)
                    else:
                        # Not adjacent - need to find path along region boundary
                        bridge = find_adjacent_path(mask, last_pos, row_start)
                        if bridge:
                            # Skip first element if it's already in path
                            start_idx = 1 if path and path[-1] == bridge[0] else 0
                            path.extend(bridge[start_idx:])
                        else:
                            # Fallback to manhattan if no adjacent path found
                            path += manhattan_path(last_pos[0], last_pos[1], row_start[0], row_start[1])
                else:
                    # First row
                    path.append(row_start)
                
                # Add rest of first segment
                for x in run[1:]:
                    path.append((x, y))
            else:
                # Connect segments within the same row
                prev_end = path[-1]
                seg_start = (run[0], y)
                # Find path between segments in same row
                bridge = find_adjacent_path(mask, prev_end, seg_start)
                if bridge:
                    # Skip first element since it's the previous end
                    path.extend(bridge[1:])
                else:
                    # Fallback to manhattan
                    path += manhattan_path(prev_end[0], prev_end[1], seg_start[0], seg_start[1])
                
                # Add rest of segment (skip first since it's already added)
                for x in run[1:]:
                    path.append((x, y))
            
            row_end = (run[-1], y)
        
        if row_end is not None:
            last_pos = row_end
        dir_lr = not dir_lr
    return path

def build_corner_side_sweep(mask: np.ndarray,
                            corner_tag: str,
                            side_tag: str,
                            anchor: Tuple[int,int]) -> List[Tuple[int,int]]:
    """
    Build a *continuous* lawnmower aligned with a chosen side from a chosen corner.
    side_tag ∈ {'LEFT','RIGHT','BOTTOM','TOP'}; corner_tag ∈ {'BL','TL','BR','TR'}.
    """
    min_x, max_x, min_y, max_y = get_region_bbox(mask)

    if side_tag == 'LEFT':
        xs = list(range(min_x, max_x + 1))
        path = sweep_columns(mask, xs, start_dir_up=(corner_tag in ('BL','TL')))
    elif side_tag == 'RIGHT':
        xs = list(range(max_x, min_x - 1, -1))
        path = sweep_columns(mask, xs, start_dir_up=(corner_tag in ('BR','TR')))
    elif side_tag == 'BOTTOM':
        ys = list(range(min_y, max_y + 1))
        path = sweep_rows(mask, ys, start_dir_lr=(corner_tag in ('BL','TL')))
    elif side_tag == 'TOP':
        ys = list(range(max_y, min_y - 1, -1))
        path = sweep_rows(mask, ys, start_dir_lr=(corner_tag in ('BR','TR')))
    else:
        raise ValueError("Invalid side_tag.")

    # Rotate to start exactly at anchor (to avoid any jump)
    rotated = rotate_path_to_exact_start(path, anchor)
    if rotated:
        return rotated
    # If anchor not in sweep due to ragged boundary (rare), connect it to the sweep start
    if path:
        ax, ay = anchor
        sx, sy = path[0]
        return [(ax, ay)] + manhattan_path(ax, ay, sx, sy) + path
    return [anchor]

# Decide which side at the chosen corner to use
def choose_side_for_corner(mask: np.ndarray,
                           corner_tag: str,
                           start_xy: Tuple[int,int]) -> str:
    """Pick between the two sides touching the corner. Heuristic:
       prefer the side whose first stripe is non-empty; if both, pick the side
       that sweeps along the *larger* span (fewer turns), tie-break by closeness to start."""
    min_x, max_x, min_y, max_y = get_region_bbox(mask)
    H, W = mask.shape
    sides_at_corner = {
        'BL': ['LEFT','BOTTOM'],
        'TL': ['LEFT','TOP'],
        'BR': ['RIGHT','BOTTOM'],
        'TR': ['RIGHT','TOP'],
    }[corner_tag]

    # Stripe sizes and first-stripe occupancy
    width  = max_x - min_x + 1
    height = max_y - min_y + 1

    def first_stripe_nonempty(side):
        if side == 'LEFT':   return any(mask[y, min_x] for y in range(H))
        if side == 'RIGHT':  return any(mask[y, max_x] for y in range(H))
        if side == 'BOTTOM': return any(mask[min_y, x] for x in range(W))
        if side == 'TOP':    return any(mask[max_y, x] for x in range(W))
        return False

    candidates = []
    for s in sides_at_corner:
        ok = first_stripe_nonempty(s)
        span = (width if s in ('LEFT','RIGHT') else height)
        candidates.append((s, ok, span))

    # 1) non-empty first stripe
    nonempty = [c for c in candidates if c[1]]
    pool = nonempty if nonempty else candidates
    # 2) larger inward span preferred (fewer stripe switches)
    max_span = max(c[2] for c in pool)
    pool2 = [c for c in pool if c[2] == max_span]
    # 3) tie-break: choose the side whose corner is closer to START (should tie anyway)
    if len(pool2) == 1:
        return pool2[0][0]
    # final tie-break: fixed order preference LEFT/BOTTOM over others for determinism
    order = {'LEFT':0,'BOTTOM':1,'RIGHT':2,'TOP':3}
    pool2.sort(key=lambda c: order[c[0]])
    return pool2[0][0]

# =========================
# Data structures
# =========================
@dataclass
class Robot:
    id: int
    path: List[Tuple[int, int]]
    idx: int = 0

    @property
    def pos(self) -> Tuple[int, int]:
        return self.path[self.idx]

    def step(self):
        if self.idx < len(self.path) - 1:
            self.idx += 1

# =========================
# Visualization
# =========================
def coverage_to_image(cv: np.ndarray) -> np.ndarray:
    img = np.ones(cv.shape, dtype=float)
    img[cv] = 0.85
    return img

def region_colors(zones: np.ndarray, alpha: float = 0.12):
    from matplotlib.cm import get_cmap
    k = zones.max() + 1
    cmap = get_cmap('tab20', k)
    rgba = cmap(zones)
    rgba[..., 3] = alpha
    return rgba

def draw_voronoi_borders(ax: plt.Axes, zones: np.ndarray, color: str = '#003366', lw: float = 1.2, alpha: float = 0.9):
    H, W = zones.shape
    for y in range(H):
        for x in range(W-1):
            if zones[y, x] != zones[y, x+1]:
                ax.plot([x+1, x+1], [y, y+1], color=color, linewidth=lw, alpha=alpha, zorder=3)
    for x in range(W):
        for y in range(H-1):
            if zones[y, x] != zones[y+1, x]:
                ax.plot([x, x+1], [y+1, y+1], color=color, linewidth=lw, alpha=alpha, zorder=3)

# =========================
# Simulation
# =========================
def sim_step():
    # Sense & discover at current pose
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y)
        discover_vn(x, y, targets, found_targets, W, H)
    # Move
    for r in robots:
        r.step()
    # Post-move sense
    for r in robots:
        x, y = r.pos
        mark_visible(covered, x, y)
        discover_vn(x, y, targets, found_targets, W, H)

def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    img = coverage_to_image(covered)
    world_img.set_data(img)
    shared_img.set_data(img)

    robot_scatter.set_offsets(np.array([[r.pos[0] + 0.5, r.pos[1] + 0.5] for r in robots]))

    for i, r in enumerate(robots):
        rem_sc, full_pts = remaining_scatters[i]
        vis_sc, _ = visited_scatters[i]
        v_pts = full_pts[:r.idx+1]
        rem_pts = full_pts[r.idx+1:]
        vis_sc.set_offsets(v_pts if len(v_pts) else np.empty((0, 2)))
        rem_sc.set_offsets(rem_pts if len(rem_pts) else np.empty((0, 2)))

    disc = list(found_targets)
    und  = list(targets - found_targets)
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
        "World — Centralized: Start → Corner+Side Anchor → Side-Aligned Continuous Lawn-mower\n"
        f"Covered: {covered.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )
    return (world_img, shared_img, robot_scatter, und_plot, disc_plot)


if __name__ == "__main__":
    # =========================
    # Parameters
    # =========================
    GRID_SIZE = 100
    N_ROBOTS   = 10
    N_TARGETS  = 20
    RANDOM_SEED = 7

    # User-defined start (clipped to grid)
    START_X, START_Y = 50, 50

    STEPS_PER_FRAME = 1
    INTERVAL_MS     = 80

    # Balanced Voronoi (capacitated k-means / power diagram) params
    MAX_ITERS_ASSIGN  = 30
    MAX_ITERS_CENTERS = 10
    LAMBDA_STEP0 = 0.1
    LAMBDA_DECAY = 0.9

    rng = np.random.default_rng(RANDOM_SEED)

    # =========================
    # Build scenario
    # =========================
    W = H = GRID_SIZE
    START_X = int(np.clip(START_X, 0, W-1))
    START_Y = int(np.clip(START_Y, 0, H-1))

    # Points at cell centers
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))

    # Balanced Voronoi partition
    labels, centers = lloyd_balanced(points, N_ROBOTS, MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN, LAMBDA_STEP0, LAMBDA_DECAY, rng)
    zones = labels.reshape(H, W)

    # Per-robot region masks; pick corner + side; anchor at nearest in-mask cell to that corner
    masks = [(zones == i) for i in range(N_ROBOTS)]
    anchors: List[Tuple[int,int]] = []
    corner_tags: List[str] = []
    side_tags: List[str] = []
    region_paths: List[List[Tuple[int,int]]] = []

    for i in range(N_ROBOTS):
        mask = masks[i]
        min_x, max_x, min_y, max_y = get_region_bbox(mask)
        # corner closest to global start
        corners = corner_candidates(min_x, max_x, min_y, max_y)
        corner_tag, (cx, cy) = min(corners.items(), key=lambda kv: abs(kv[1][0]-START_X)+abs(kv[1][1]-START_Y))
        corner_tags.append(corner_tag)

        # side at that corner
        side_tag = choose_side_for_corner(mask, corner_tag, (START_X, START_Y))
        side_tags.append(side_tag)

        # in-region anchor nearest to chosen geometric corner
        ax_cell, ay_cell = find_anchor_near_point(mask, cx, cy)
        anchors.append((ax_cell, ay_cell))

        # build continuous sweep aligned with that side
        rp = build_corner_side_sweep(mask, corner_tag, side_tag, (ax_cell, ay_cell))
        region_paths.append(rp)

    # Build full continuous paths: Start → Transit → Anchor → Sweep
    full_paths: List[List[Tuple[int,int]]] = []
    for i in range(N_ROBOTS):
        anchor = anchors[i]
        rp = region_paths[i]
        transit = manhattan_path(START_X, START_Y, anchor[0], anchor[1])
        path = [(START_X, START_Y)]
        path = concat_paths(path, transit)
        path = concat_paths(path, rp)
        full_paths.append(path)

    robots = [Robot(i, full_paths[i]) for i in range(N_ROBOTS)]

    # Shared state (centralized)
    covered = np.zeros((H, W), dtype=bool)
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    found_targets: Set[Tuple[int, int]] = set()


    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_shared = fig.add_subplot(gs[0, 1])

    zone_rgba = region_colors(zones, alpha=0.12)
    ax_world.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
    world_img = ax_world.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0, vmax=1)
    draw_voronoi_borders(ax_world, zones, color='#003366', lw=1.2, alpha=0.9)

    ax_world.set_title("World — Centralized: Start → Corner+Side Anchor → Side-Aligned Continuous Lawn-mower")
    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')
    ax_world.set_xticks(np.arange(0, W+1, 10)); ax_world.set_yticks(np.arange(0, H+1, 10))
    ax_world.set_xticks(np.arange(0, W+1, 1), minor=True); ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    # Robots & path overlays
    robot_scatter = ax_world.scatter(
        [r.pos[0] + 0.5 for r in robots],
        [r.pos[1] + 0.5 for r in robots],
        s=40, marker='o', c='k', zorder=4
    )

    remaining_scatters, visited_scatters = [], []
    for r in robots:
        full_pts = np.array([[x + 0.5, y + 0.5] for (x, y) in r.path])
        rem_sc = ax_world.scatter(full_pts[:, 0], full_pts[:, 1], s=6, marker='s',
                                facecolors='none', edgecolors='0.35', alpha=0.14, linewidths=0.45, zorder=1)
        vis_sc = ax_world.scatter([], [], s=10, marker='s',
                                facecolors='none', edgecolors='0.2', alpha=0.55, linewidths=0.6, zorder=2)
        remaining_scatters.append((rem_sc, full_pts))
        visited_scatters.append((vis_sc, []))

    # Start & targets
    ax_world.scatter([START_X + 0.5], [START_Y + 0.5], s=60, marker='*', c='gold', edgecolors='k', zorder=5)
    if targets:
        tx, ty = zip(*targets)
    else:
        tx, ty = [], []
    ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty], s=20, marker='x', c='r', alpha=0.9, zorder=4)

    # Shared panel
    ax_shared.imshow(zone_rgba, origin='lower', extent=[0, W, 0, H])
    shared_img = ax_shared.imshow(coverage_to_image(covered), origin='lower', extent=[0, W, 0, H], vmin=0, vmax=1)
    draw_voronoi_borders(ax_shared, zones, color='#003366', lw=1.2, alpha=0.9)

    ax_shared.set_title("Shared Map — Global Knowledge")
    ax_shared.set_xlim(0, W); ax_shared.set_ylim(0, H); ax_shared.set_aspect('equal', adjustable='box')
    ax_shared.set_xticks(np.arange(0, W+1, 10)); ax_shared.set_yticks(np.arange(0, H+1, 10))
    ax_shared.set_xticks(np.arange(0, W+1, 1), minor=True); ax_shared.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_shared.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_shared.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    disc_plot = ax_shared.scatter([], [], s=25, marker='o', facecolors='none',
                                edgecolors='g', linewidths=1.5, label='Discovered')
    und_plot  = ax_shared.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                                s=18, marker='x', c='r', label='Undiscovered')
    ax_shared.legend(loc='upper right', fontsize=8, frameon=False)

    # =========================
    # Run
    # =========================
    anim = FuncAnimation(fig, update, frames=200000, interval=INTERVAL_MS, blit=False)
    plt.show()