import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

# =========================================
# Parameters
# =========================================
GRID_SIZE = 100
N_ROBOTS   = 6
N_TARGETS  = 30
RANDOM_SEED = 7

STEPS_PER_FRAME = 1
INTERVAL_MS     = 65

# Pheromones
PHER_EXP_DEPOSIT_SELF   = 1.0    # exploration deposit at current cell
PHER_EXP_DEPOSIT_AHEAD1 = 0.6    # ahead streak (1 cell)
PHER_EXP_DEPOSIT_AHEAD2 = 0.3    # ahead streak (2 cells)
TAU_EXP   = 50.0                 # fast decay (exploration)
TAU_ATTR  = 200.0                # slow decay (attraction)
TAU_EVENT = 300.0                # for hazards/dead-ends if used

PHER_MIN  = 1e-6

# Motion biases (higher -> stronger effect)
W_UNCOVERED  = 2.0   # prefer cells not yet seen by THIS robot
W_FRONTIER   = 1.5   # prefer moving toward nearest private frontier
W_EXP_AVOID  = 2.0   # avoid exploration pheromone (inverse gradient)
W_ATTR_SEEK  = 2.0   # seek attraction pheromone (toward targets)
EPS_NOISE    = 0.05  # small exploration noise to avoid ties

# Frontier search parameters
FRONTIER_SEARCH_RADIUS = 20  # how far we look when estimating nearest frontier (Manhattan)
rng = np.random.default_rng(RANDOM_SEED)

# =========================================
# Utilities
# =========================================
def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    idx = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in idx)

def neighbors_vn(x: int, y: int, W: int, H: int) -> List[Tuple[int,int]]:
    out = [(x, y)]
    if y-1 >= 0: out.append((x, y-1))
    if y+1 < H:  out.append((x, y+1))
    if x-1 >= 0: out.append((x-1, y))
    if x+1 < W:  out.append((x+1, y))
    return out

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int):
    """Mark current cell + VN neighbors True on provided bool grid."""
    H, W = grid_bool.shape
    grid_bool[y, x] = True
    if y-1 >= 0: grid_bool[y-1, x] = True
    if y+1 < H:  grid_bool[y+1, x] = True
    if x-1 >= 0: grid_bool[y, x-1] = True
    if x+1 < W:  grid_bool[y, x+1] = True

def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_vn(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

def is_frontier(cell_x: int, cell_y: int, local_map: np.ndarray) -> bool:
    """A frontier cell is a KNOWN cell having at least one UNKNOWN VN neighbor."""
    H, W = local_map.shape
    if not local_map[cell_y, cell_x]:
        return False
    if cell_y-1 >= 0 and not local_map[cell_y-1, cell_x]: return True
    if cell_y+1 < H and not local_map[cell_y+1, cell_x]:  return True
    if cell_x-1 >= 0 and not local_map[cell_y, cell_x-1]: return True
    if cell_x+1 < W and not local_map[cell_y, cell_x+1]:  return True
    return False

def nearest_frontier_distance(x: int, y: int, local_map: np.ndarray, radius: int) -> int:
    """Approximate Manhattan distance to nearest frontier within radius; returns radius+1 if none."""
    H, W = local_map.shape
    best = radius + 1
    for dy in range(-radius, radius+1):
        yy = y + dy
        if yy < 0 or yy >= H: continue
        span = radius - abs(dy)
        for dx in range(-span, span+1):
            xx = x + dx
            if xx < 0 or xx >= W: continue
            if is_frontier(xx, yy, local_map):
                d = abs(dx) + abs(dy)
                if d < best: best = d
    return best

# =========================================
# Robot
# =========================================
@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray         # private bool map
    last_move: Optional[Tuple[int,int]] = None

    # def choose_move(self, pher_exp: np.ndarray, pher_attr: np.ndarray) -> Tuple[int, int]:
    #     H, W = pher_exp.shape
    #     cands: List[Tuple[int,int]] = []
    #     if self.y-1 >= 0: cands.append((self.x, self.y-1))
    #     if self.y+1 < H:  cands.append((self.x, self.y+1))
    #     if self.x-1 >= 0: cands.append((self.x-1, self.y))
    #     if self.x+1 < W:  cands.append((self.x+1, self.y))
    #     if not cands:
    #         return (self.x, self.y)

    #     # Score each candidate with directed exploration:
    #     # + prefer local-UNcovered
    #     # + prefer proximity to frontier (smaller distance -> larger score)
    #     # + avoid exploration pheromone (lower exp better)
    #     # + seek attraction pheromone (higher attr better)
    #     scores = []
    #     for (nx, ny) in cands:
    #         # Private map terms
    #         uncov_bonus = (not self.local_covered[ny, nx])
    #         # approximate frontier 'pull' using distance from candidate to nearest frontier
    #         d_front = nearest_frontier_distance(nx, ny, self.local_covered, FRONTIER_SEARCH_RADIUS)
    #         front_term = (FRONTIER_SEARCH_RADIUS + 1 - d_front) / (FRONTIER_SEARCH_RADIUS + 1)  # in [0,1]

    #         # Pheromone terms
    #         p_exp  = max(pher_exp[ny, nx], 0.0)
    #         p_attr = max(pher_attr[ny, nx], 0.0)
    #         exp_term  = - p_exp / (1.0 + p_exp)   # decreasing in p_exp
    #         attr_term =   p_attr / (1.0 + p_attr) # increasing in p_attr

    #         # Combine with weights
    #         s = (W_UNCOVERED * (1.0 if uncov_bonus else 0.0) +
    #              W_FRONTIER  * front_term +
    #              W_EXP_AVOID * exp_term +
    #              W_ATTR_SEEK * attr_term +
    #              EPS_NOISE * rng.random())
    #         scores.append(s)

    #     scores = np.array(scores, dtype=float)
    #     # softmax-ish selection to keep some stochasticity
    #     # if all equal, choose uniformly
    #     if np.allclose(scores, scores[0]):
    #         idx = rng.choice(len(cands))
    #         return cands[idx]
    #     # stable softmax
    #     z = scores - scores.max()
    #     probs = np.exp(z)
    #     probs /= probs.sum()
    #     idx = rng.choice(len(cands), p=probs)
    #     return cands[idx]
    
    def choose_move(self, pher_exp: np.ndarray, pher_attr: np.ndarray) -> Tuple[int, int]:
        H, W = pher_exp.shape
        cands: List[Tuple[int,int]] = []
        if self.y-1 >= 0: cands.append((self.x, self.y-1))
        if self.y+1 < H:  cands.append((self.x, self.y+1))
        if self.x-1 >= 0: cands.append((self.x-1, self.y))
        if self.x+1 < W:  cands.append((self.x+1, self.y))
        if not cands:
            return (self.x, self.y)

        # HARD no-revisit: first try only neighbors this robot hasn't covered (its own local VN-sensed map)
        fresh = [(nx, ny) for (nx, ny) in cands if not self.local_covered[ny, nx]]
        pool = fresh if fresh else cands  # fall back ONLY if boxed in by previously-covered neighbors

        # Score pool with directed exploration (frontier + pheromones). Keeps some stochasticity within allowed set.
        scores = []
        for (nx, ny) in pool:
            # frontier bias (toward nearest frontier of THIS robot's map)
            d_front = nearest_frontier_distance(nx, ny, self.local_covered, FRONTIER_SEARCH_RADIUS)
            front_term = (FRONTIER_SEARCH_RADIUS + 1 - d_front) / (FRONTIER_SEARCH_RADIUS + 1)  # in [0,1]

            # pheromone terms
            p_exp  = max(pher_exp[ny, nx], 0.0)
            p_attr = max(pher_attr[ny, nx], 0.0)
            exp_term  = - p_exp / (1.0 + p_exp)   # avoid exploration pheromone
            attr_term =   p_attr / (1.0 + p_attr) # seek attraction pheromone

            # uncovered boost (within pool, many will be fresh; if we're in fallback, some will be covered)
            uncov_boost = 1.0 if not self.local_covered[ny, nx] else 0.0

            s = (W_UNCOVERED * uncov_boost +
                W_FRONTIER  * front_term +
                W_EXP_AVOID * exp_term +
                W_ATTR_SEEK * attr_term +
                EPS_NOISE * rng.random())
            scores.append(s)

        scores = np.array(scores, dtype=float)
        if np.allclose(scores, scores[0]):
            idx = rng.choice(len(pool))
            return pool[idx]
        z = scores - scores.max()
        probs = np.exp(z); probs /= probs.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[idx]

    def step(self, pher_exp: np.ndarray, pher_attr: np.ndarray):
        nx, ny = self.choose_move(pher_exp, pher_attr)
        self.last_move = (np.clip(nx - self.x, -1, 1), np.clip(ny - self.y, -1, 1))
        self.x, self.y = nx, ny

# =========================================
# World setup
# =========================================
W = H = GRID_SIZE
all_cells = [(x, y) for x in range(W) for y in range(H)]
spawn_idx = rng.choice(len(all_cells), size=N_ROBOTS, replace=False)
spawns = [all_cells[i] for i in spawn_idx]
# robots = [Robot(i, x, y, np.zeros((H, W), dtype=bool)) for i, (x, y) in enumerate(spawns)]
robots = [Robot(i, 50, 50, np.zeros((H, W), dtype=bool)) for i, (x, y) in enumerate(spawns)]

targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
found_targets: Set[Tuple[int,int]] = set()

# Visualization convenience (union coverage—not shared by robots)
covered_union = np.zeros((H, W), dtype=bool)

# Pheromone layers
pher_exp  = np.zeros((H, W), dtype=float)  # exploration (pink)
pher_attr = np.zeros((H, W), dtype=float)  # attraction  (cyan/teal)
# Optional event layers (orange/yellow)
pher_hazard  = np.zeros((H, W), dtype=float)
pher_deadend = np.zeros((H, W), dtype=float)

# Decay factors
EXP_DECAY   = np.exp(-1.0 / TAU_EXP)
ATTR_DECAY  = np.exp(-1.0 / TAU_ATTR)
EVENT_DECAY = np.exp(-1.0 / TAU_EVENT)

# =========================================
# Visualization helpers
# =========================================
def coverage_to_image(cv: np.ndarray) -> np.ndarray:
    img = np.ones(cv.shape, dtype=float)
    img[cv] = 0.85
    return img

def layer_to_rgba(layer: np.ndarray, rgb: Tuple[float,float,float], alpha_scale: float) -> np.ndarray:
    vmax = max(np.percentile(layer, 95), PHER_MIN)
    norm = np.clip(layer / vmax, 0.0, 1.0) if vmax > 0 else np.zeros_like(layer)
    rgba = np.zeros((layer.shape[0], layer.shape[1], 4), dtype=float)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = norm * alpha_scale
    return rgba

# =========================================
# Matplotlib layout
# =========================================
fig = plt.figure(figsize=(12.8, 6.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
ax_world = fig.add_subplot(gs[0, 0])
ax_obs   = fig.add_subplot(gs[0, 1])

# Left: world (coverage + pheromones + robots)
img_cov  = ax_world.imshow(coverage_to_image(covered_union), origin='lower', extent=[0, W, 0, H], vmin=0, vmax=1, zorder=0)
img_exp  = ax_world.imshow(layer_to_rgba(pher_exp,  (1.0, 0.2, 0.6), 0.35), origin='lower', extent=[0, W, 0, H], zorder=1)  # pink
img_attr = ax_world.imshow(layer_to_rgba(pher_attr, (0.1, 0.9, 0.9), 0.35), origin='lower', extent=[0, W, 0, H], zorder=1)  # cyan
img_haz  = ax_world.imshow(layer_to_rgba(pher_hazard,  (1.0, 0.55, 0.0), 0.30), origin='lower', extent=[0, W, 0, H], zorder=1)  # orange
img_dead = ax_world.imshow(layer_to_rgba(pher_deadend, (1.0, 0.9, 0.1), 0.25), origin='lower', extent=[0, W, 0, H], zorder=1)  # yellow

# Grid
ax_world.set_xticks(np.arange(0, W+1, 10)); ax_world.set_yticks(np.arange(0, H+1, 10))
ax_world.set_xticks(np.arange(0, W+1, 1), minor=True); ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

# Robots & targets
robot_scat = ax_world.scatter([r.x + 0.5 for r in robots], [r.y + 0.5 for r in robots], s=40, marker='o', c='k', zorder=3)
if targets:
    tx, ty = zip(*targets)
else:
    tx, ty = [], []
ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty], s=20, marker='x', c='r', alpha=0.9, zorder=4)

ax_world.set_title("World — Stigmergy v2 (Directed Exploration: Frontier + Inverse-Gradient + Multi-Pheromone)")
ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')

# Right: observer panel (not shared by robots)
img_exp_obs  = ax_obs.imshow(layer_to_rgba(pher_exp,  (1.0, 0.2, 0.6), 0.45), origin='lower', extent=[0, W, 0, H], zorder=0)
img_attr_obs = ax_obs.imshow(layer_to_rgba(pher_attr, (0.1, 0.9, 0.9), 0.45), origin='lower', extent=[0, W, 0, H], zorder=0)
img_haz_obs  = ax_obs.imshow(layer_to_rgba(pher_hazard,  (1.0, 0.55, 0.0), 0.35), origin='lower', extent=[0, W, 0, H], zorder=0)
img_dead_obs = ax_obs.imshow(layer_to_rgba(pher_deadend, (1.0, 0.9, 0.1), 0.30), origin='lower', extent=[0, W, 0, H], zorder=0)

und_plot = ax_obs.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty], s=18, marker='x', c='r', label='Undiscovered', zorder=2)
disc_plot = ax_obs.scatter([], [], s=25, marker='o', facecolors='none', edgecolors='g', linewidths=1.5, label='Discovered', zorder=2)

ax_obs.set_xticks(np.arange(0, W+1, 10)); ax_obs.set_yticks(np.arange(0, H+1, 10))
ax_obs.set_xticks(np.arange(0, W+1, 1), minor=True); ax_obs.set_yticks(np.arange(0, H+1, 1), minor=True)
ax_obs.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
ax_obs.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
ax_obs.set_title("Observer — Pheromone Layers (robots do NOT share maps)")
ax_obs.set_xlim(0, W); ax_obs.set_ylim(0, H); ax_obs.set_aspect('equal', adjustable='box')
ax_obs.legend(loc='upper right', fontsize=8, frameon=False)

# =========================================
# Pheromone deposition helpers
# =========================================
def deposit_exploration_streak(pher_exp: np.ndarray, x: int, y: int, move: Optional[Tuple[int,int]]):
    """Deposit at current cell and a faint directional streak ahead (1–2 cells) to encode heading."""
    H, W = pher_exp.shape
    pher_exp[y, x] += PHER_EXP_DEPOSIT_SELF
    if move is None: return
    dx, dy = move
    if dx == 0 and dy == 0: return
    x1, y1 = x + dx, y + dy
    if 0 <= x1 < W and 0 <= y1 < H:
        pher_exp[y1, x1] += PHER_EXP_DEPOSIT_AHEAD1
        x2, y2 = x1 + dx, y1 + dy
        if 0 <= x2 < W and 0 <= y2 < H:
            pher_exp[y2, x2] += PHER_EXP_DEPOSIT_AHEAD2

def deposit_attraction(pher_attr: np.ndarray, cells: List[Tuple[int,int]], amount: float = 3.0):
    for (x, y) in cells:
        if 0 <= x < pher_attr.shape[1] and 0 <= y < pher_attr.shape[0]:
            pher_attr[y, x] += amount

# =========================================
# Simulation step
# =========================================
def sim_step():
    global pher_exp, pher_attr, pher_hazard, pher_deadend

    # Evaporation
    pher_exp  *= EXP_DECAY;   pher_exp[pher_exp < PHER_MIN]   = 0.0
    pher_attr *= ATTR_DECAY;  pher_attr[pher_attr < PHER_MIN] = 0.0
    pher_hazard  *= EVENT_DECAY; pher_deadend *= EVENT_DECAY
    pher_hazard[pher_hazard < PHER_MIN] = 0.0
    pher_deadend[pher_deadend < PHER_MIN] = 0.0

    # Sense / cover / discover; deposit exploration + attraction when targets appear in VN
    for r in robots:
        # Update private & union coverage from current pose
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_union,   r.x, r.y)
        # Discover targets; if seen, drop attraction marker at the target cell(s)
        before = len(found_targets)
        # collect VN targets for attraction deposition even if already discovered (refresh signal)
        vn_targets = []
        for (nx, ny) in neighbors_vn(r.x, r.y, W, H):
            if (nx, ny) in targets:
                found_targets.add((nx, ny))
                vn_targets.append((nx, ny))
        if vn_targets:
            deposit_attraction(pher_attr, vn_targets, amount=4.0)

        # Deposit exploration streak encoding heading (uses last_move from previous step)
        deposit_exploration_streak(pher_exp, r.x, r.y, r.last_move)

    # Move robots using directed policy
    for r in robots:
        r.step(pher_exp, pher_attr)

    # Post-move sensing + light deposit to leave breadcrumbs at arrival
    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_union,   r.x, r.y)
        # re-check VN for targets and refresh attraction slightly
        vn_targets = []
        for (nx, ny) in neighbors_vn(r.x, r.y, W, H):
            if (nx, ny) in targets:
                found_targets.add((nx, ny))
                vn_targets.append((nx, ny))
        if vn_targets:
            deposit_attraction(pher_attr, vn_targets, amount=1.5)
        # slight local breadcrumb
        pher_exp[r.y, r.x] += 0.25

# =========================================
# Animation update
# =========================================
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    # Update layers
    img_cov.set_data(coverage_to_image(covered_union))
    img_exp.set_data(layer_to_rgba(pher_exp,  (1.0, 0.2, 0.6), 0.35))
    img_attr.set_data(layer_to_rgba(pher_attr, (0.1, 0.9, 0.9), 0.35))
    img_haz.set_data(layer_to_rgba(pher_hazard,  (1.0, 0.55, 0.0), 0.30))
    img_dead.set_data(layer_to_rgba(pher_deadend, (1.0, 0.9, 0.1), 0.25))

    # Robots
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))

    # Observer overlays
    img_exp_obs.set_data(layer_to_rgba(pher_exp,  (1.0, 0.2, 0.6), 0.45))
    img_attr_obs.set_data(layer_to_rgba(pher_attr, (0.1, 0.9, 0.9), 0.45))
    img_haz_obs.set_data(layer_to_rgba(pher_hazard,  (1.0, 0.55, 0.0), 0.35))
    img_dead_obs.set_data(layer_to_rgba(pher_deadend, (1.0, 0.9, 0.1), 0.30))

    # Targets
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
        "World — Stigmergy v2 (Directed Exploration: Frontier + Inverse-Gradient + Multi-Pheromone)\n"
        f"Covered (union): {covered_union.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )
    return (img_cov, img_exp, img_attr, robot_scat, img_exp_obs, img_attr_obs, und_plot, disc_plot)

# =========================================
# Run
# =========================================
anim = FuncAnimation(fig, update, frames=200000, interval=INTERVAL_MS, blit=False)
plt.show()
