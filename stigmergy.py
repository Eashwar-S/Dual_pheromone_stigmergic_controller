import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

from simulation import FrameWriter, compute_fps, make_writer, run_animation


# -----------------------------
# Utilities
# -----------------------------
def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

def neighbors_vn(x: int, y: int, W: int, H: int) -> List[Tuple[int,int]]:
    out = [(x, y)]
    if y-1 >= 0: out.append((x, y-1))
    if y+1 < H:  out.append((x, y+1))
    if x-1 >= 0: out.append((x-1, y))
    if x+1 < W:  out.append((x+1, y))
    return out

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int):
    """Mark current cell + VN neighbors True on the provided boolean grid."""
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

# -----------------------------
# Robot with PRIVATE local map  (NEW)
# -----------------------------
@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray  # (H,W) bool, private per-robot map
    last_move: Optional[Tuple[int,int]] = None

    def choose_move(self, pher: np.ndarray) -> Tuple[int,int]:
        """Biased random step among VN neighbors (no 'stay').
        Primary bias: prefer cells this robot has NOT covered on its own map (local_covered=False).
        Secondary bias: avoid higher pheromone.
        """
        H, W = pher.shape
        candidates: List[Tuple[int,int]] = []
        if self.y-1 >= 0: candidates.append((self.x, self.y-1))
        if self.y+1 < H:  candidates.append((self.x, self.y+1))
        if self.x-1 >= 0: candidates.append((self.x-1, self.y))
        if self.x+1 < W:  candidates.append((self.x+1, self.y))

        if not candidates:
            return (self.x, self.y)

        # Partition candidates into local-uncovered vs local-covered (NEW)
        uncov = [(nx, ny) for (nx, ny) in candidates if not self.local_covered[ny, nx]]
        pool = uncov if len(uncov) > 0 else candidates  # prefer unexplored; fallback if none

        # Build pheromone-avoidance weights, with optional slight bonus for globally uncovered effect removed
        weights = []
        for (nx, ny) in pool:
            p = max(pher[ny, nx], 0.0)
            desirability = np.exp(-BIAS_ALPHA * (p / (1.0 + p)))
            # keep a small bias toward cells this robot hasn't covered (if we’re in fallback pool==candidates, some may be covered)
            if not self.local_covered[ny, nx]:
                desirability *= UNCOVERED_BONUS
            weights.append(desirability)

        w = np.array(weights, dtype=float)
        if np.all(w <= 0):  # numerical fallback
            w = np.ones_like(w)
        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[idx]

    def step(self, pher: np.ndarray):
        nx, ny = self.choose_move(pher)
        self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny

# -----------------------------
# Visualization helpers
# -----------------------------
def coverage_to_image(cv: np.ndarray) -> np.ndarray:
    img = np.ones(cv.shape, dtype=float)
    img[cv] = 0.85
    return img

def pheromone_to_rgba(ph: np.ndarray, alpha_scale: float = 0.35) -> np.ndarray:
    vmax = max(np.percentile(ph, 95), PHER_MIN)
    norm = np.clip(ph / vmax, 0.0, 1.0)
    rgba = np.zeros((ph.shape[0], ph.shape[1], 4), dtype=float)
    rgba[..., 0] = 1.0   # pink
    rgba[..., 1] = 0.2
    rgba[..., 2] = 0.6
    rgba[..., 3] = norm * alpha_scale
    return rgba

# -----------------------------
# Simulation step
# -----------------------------
def sim_step():
    global pher

    # Evaporate pheromone
    pher *= np.exp(-1.0 / TAU_DECAY)
    pher[pher < PHER_MIN] = 0.0

    # Sense/cover/discover; deposit; advance
    for r in robots:
        # Mark on robot's **private** map (NEW) and global viz
        mark_visible_bool(r.local_covered, r.x, r.y)  # private
        mark_visible_bool(covered_global, r.x, r.y)   # visualization union
        discover_vn(r.x, r.y, targets, found_targets, W, H)

        # Deposit at current cell
        pher[r.y, r.x] += PHER_DEPOSIT

    # Move robots using their local maps (NEW)
    for r in robots:
        r.step(pher)

    # Post-move sensing & a small extra deposit
    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_global, r.x, r.y)
        discover_vn(r.x, r.y, targets, found_targets, W, H)
        pher[r.y, r.x] += 0.3 * PHER_DEPOSIT

# -----------------------------
# Animation update
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    world_cov_img.set_data(coverage_to_image(covered_global))
    world_pher_img.set_data(pheromone_to_rgba(pher))
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
    obs_pher_img.set_data(pheromone_to_rgba(pher))

    discovered = list(found_targets)
    undiscovered = list(targets - found_targets)
    if discovered:
        dx, dy = zip(*discovered)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    else:
        disc_plot.set_offsets(np.empty((0, 2)))
    if undiscovered:
        ux, uy = zip(*undiscovered)
        und_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
    else:
        und_plot.set_offsets(np.empty((0, 2)))

    ax_world.set_title(
        "World — Stigmergy (Local Maps + Random Walk + Pheromone)\n"
        f"Covered (union): {covered_global.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )

    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat, obs_pher_img, und_plot, disc_plot)

if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    GRID_SIZE = 25
    N_ROBOTS = 8
    N_TARGETS = 30
    RANDOM_SEED = 7
    STEPS_PER_FRAME = 1
    INTERVAL_MS = 80

    # Stigmergy / pheromone
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 60.0
    PHER_MIN = 1e-6
    BIAS_ALPHA = 2.5          # avoid pheromone strength
    UNCOVERED_BONUS = 10.0     # (kept) slight bonus for unexplored
    rng = np.random.default_rng(RANDOM_SEED)


    FPS = compute_fps(INTERVAL_MS)
    writer = make_writer(INTERVAL_MS, title="Stigmergy - random walk", artist="you")
    dir = "output_frames/stigmergy_random_walk/"
    frame_writer = FrameWriter(dir)

    # -----------------------------
    # World setup
    # -----------------------------
    W = H = GRID_SIZE
    robot_starting_x = W // 2
    robot_starting_y = H // 2
    # Robots spawn  
    all_cells = [(x, y) for x in range(W) for y in range(H)]
    spawn_idx = rng.choice(len(all_cells), size=N_ROBOTS, replace=False)
    spawn_positions = [all_cells[i] for i in spawn_idx]
    # robots = [Robot(i, x, y, local_covered=np.zeros((H, W), dtype=bool)) for i, (x, y) in enumerate(spawn_positions)]
    robots = [Robot(i, robot_starting_x, robot_starting_y, local_covered=np.zeros((H, W), dtype=bool)) for i in range(N_ROBOTS)]

    # Targets & state
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    found_targets: Set[Tuple[int,int]] = set()

    # Global (for visualization only — NOT shared by robots)
    covered_global = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)
    DECAY_FACTOR = np.exp(-1.0 / TAU_DECAY)

    # -----------------------------
    # Matplotlib layout
    # -----------------------------
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs   = fig.add_subplot(gs[0, 1])

    world_cov_img = ax_world.imshow(coverage_to_image(covered_global), origin='lower',
                                    extent=[0, W, 0, H], vmin=0.0, vmax=1.0, zorder=0)
    world_pher_img = ax_world.imshow(pheromone_to_rgba(pher), origin='lower',
                                    extent=[0, W, 0, H], zorder=1)

    # Faint grid
    ax_world.set_xticks(np.arange(0, W+1, 10)); ax_world.set_yticks(np.arange(0, H+1, 10))
    ax_world.set_xticks(np.arange(0, W+1, 1), minor=True); ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    robot_scat = ax_world.scatter([r.x + 0.5 for r in robots], [r.y + 0.5 for r in robots],
                                s=40, marker='o', c='k', zorder=3)

    # Targets
    if targets:
        tx, ty = zip(*targets)
    else:
        tx, ty = [], []
    ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)

    ax_world.set_title("World — Stigmergy (Local Maps + Random Walk + Pheromone)")
    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')

    # Observer view (not shared by robots)
    obs_pher_img = ax_obs.imshow(pheromone_to_rgba(pher), origin='lower',
                                extent=[0, W, 0, H], zorder=0)
    und_plot = ax_obs.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                            s=18, marker='x', c='r', label='Undiscovered', zorder=2)
    disc_plot = ax_obs.scatter([], [], s=25, marker='o', facecolors='none',
                            edgecolors='g', linewidths=1.5, label='Discovered', zorder=2)

    ax_obs.set_xticks(np.arange(0, W+1, 10)); ax_obs.set_yticks(np.arange(0, H+1, 10))
    ax_obs.set_xticks(np.arange(0, W+1, 1), minor=True); ax_obs.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_obs.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_obs.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    ax_obs.set_title("Observer — Pheromone Field (Robots don't share maps)")
    ax_obs.set_xlim(0, W); ax_obs.set_ylim(0, H); ax_obs.set_aspect('equal', adjustable='box')
    ax_obs.legend(loc='upper right', fontsize=8, frameon=False)

    

    # -----------------------------
    # Run
    # -----------------------------
    anim = run_animation(fig, update, frames=200000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
