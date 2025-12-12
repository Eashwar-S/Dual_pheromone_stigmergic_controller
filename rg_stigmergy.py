import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from pathlib import Path
import random
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

def neighbors_vn_r(x: int, y: int, W: int, H: int, r: int = 5):
    out = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:  # VN metric (Manhattan distance)
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out.append((nx, ny))
    return out

def would_gain_coverage(nx: int, ny: int, local_covered: np.ndarray, W: int, H: int, r: int) -> bool:
    """Return True iff moving to (nx, ny) would mark at least one new cell within radius r."""
    for (cx, cy) in neighbors_vn_r(nx, ny, W, H, r=r):
        if not local_covered[cy, cx]:
            return True
    return False

def coverage_gain_count(nx: int, ny: int, local_covered: np.ndarray, W: int, H: int, r: int) -> int:
    """How many currently-uncovered cells within radius r would be newly covered if at (nx, ny)?"""
    cnt = 0
    for (cx, cy) in neighbors_vn_r(nx, ny, W, H, r=r):
        if not local_covered[cy, cx]:
            cnt += 1
    return cnt


# def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int):
#     """Mark current cell + VN neighbors True on the provided boolean grid."""
#     H, W = grid_bool.shape
#     grid_bool[y, x] = True
#     if y-1 >= 0: grid_bool[y-1, x] = True
#     if y+1 < H:  grid_bool[y+1, x] = True
#     if x-1 >= 0: grid_bool[y, x-1] = True
#     if x+1 < W:  grid_bool[y, x+1] = True

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int, r: int = 5):
    H, W = grid_bool.shape
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    grid_bool[ny, nx] = True


def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_vn_r(x, y, W, H):
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
    failed: bool = False  # <-- NEW

    # def choose_move(self, red_pher: np.ndarray, green_pher: np.ndarray) -> Tuple[int,int]:
    #     """Biased random step: avoid red (exploration), attracted to green (targets)."""
    #     H, W = red_pher.shape
    #     candidates: List[Tuple[int,int]] = []
    #     if self.y-1 >= 0: candidates.append((self.x, self.y-1))
    #     if self.y+1 < H:  candidates.append((self.x, self.y+1))
    #     if self.x-1 >= 0: candidates.append((self.x-1, self.y))
    #     if self.x+1 < W:  candidates.append((self.x+1, self.y))
    #     if not candidates:
    #         return (self.x, self.y)

    #     # Prefer locally-uncovered cells
    #     uncov = [(nx, ny) for (nx, ny) in candidates if not self.local_covered[ny, nx]]
    #     pool = uncov if len(uncov) > 0 else candidates
    #     weights = []
    #     for (nx, ny) in pool:
    #         red_p = max(red_pher[ny, nx], 0.0)
    #         green_p = max(green_pher[ny, nx], 0.0)
            
    #         # REPULSION from red (exploration pheromone)
    #         red_factor = np.exp(-BIAS_ALPHA_RED * (red_p / (1.0 + red_p)))
            
    #         # ATTRACTION to green (target pheromone)
    #         green_factor = 1.0 + BIAS_ALPHA_GREEN * (green_p / (1.0 + green_p))
            
    #         desirability = red_factor * green_factor
            
    #         # Bonus for unexplored
    #         if not self.local_covered[ny, nx]:
    #             desirability *= UNCOVERED_BONUS
    #         weights.append(desirability)
        
    #     w = np.array(weights, dtype=float)
    #     if np.all(w <= 0):
    #         w = np.ones_like(w)
    #     probs = w / w.sum()
    #     idx = rng.choice(len(pool), p=probs)
    #     return pool[idx]

    def choose_move(self, red_pher: np.ndarray, green_pher: np.ndarray) -> Tuple[int, int]:
        # 4-connected neighbors
        H, W = self.local_covered.shape
        candidates: List[Tuple[int, int]] = []
        if self.y - 1 >= 0: candidates.append((self.x, self.y - 1))
        if self.y + 1 < H:  candidates.append((self.x, self.y + 1))
        if self.x - 1 >= 0: candidates.append((self.x - 1, self.y))
        if self.x + 1 < W:  candidates.append((self.x + 1, self.y))
        if not candidates:
            return (self.x, self.y)

        # Coverage gain for each neighbor (how much new area within radius r would be covered)
        r_cov = COVER_RADIUS_FOR_MOVE  # must match sensing radius used elsewhere
        gains = [(nx, ny, coverage_gain_count(nx, ny, self.local_covered, W, H, r_cov))
                for (nx, ny) in candidates]

        # If no neighbor adds any new coverage, stay put (avoids redundancy)
        max_gain = max(g for _, _, g in gains) if gains else 0
        if max_gain <= 0:
            return (self.x, self.y)

        # --- Stochastic selection biased by coverage gain + pheromones ---
        # Boltzmann weight on coverage gain to restore randomness while preferring higher gain
        beta = 1.0  # higher => greedier (beta=0 => uniform)
        desirabilities = []
        pool = []
        for (nx, ny, g) in gains:
            if g <= 0:
                continue  # only consider moves that add new coverage

            # Coverage preference (softmax over gain)
            coverage_term = float(np.exp(beta * g))

            # REPULSION from red (exploration pheromone)
            red_p = max(red_pher[ny, nx], 0.0)
            red_factor = np.exp(-BIAS_ALPHA_RED * (red_p / (1.0 + red_p)))

            # ATTRACTION to green (target pheromone)
            green_p = max(green_pher[ny, nx], 0.0)
            green_factor = 1.0 + BIAS_ALPHA_GREEN * (green_p / (1.0 + green_p))

            desirability = coverage_term * red_factor * green_factor

            # Bonus for stepping onto an as-yet-unseen center cell
            if not self.local_covered[ny, nx]:
                desirability *= UNCOVERED_BONUS

            pool.append((nx, ny))
            desirabilities.append(desirability)

        if not pool:
            return (self.x, self.y)

        w = np.asarray(desirabilities, dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones_like(w)
        probs = w / w.sum()

        idx = rng.choice(len(pool), p=probs)
        return pool[idx]
    

    def step(self, red_pher: np.ndarray, green_pher: np.ndarray):
        if self.failed:
            return  # <-- NEW: freeze in place once failed
        nx, ny = self.choose_move(red_pher, green_pher)
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

def green_pheromone_to_rgba(ph: np.ndarray, alpha_scale: float = 0.5) -> np.ndarray:
    """Green pheromone for target signals - more visible."""
    vmax = max(np.percentile(ph, 95), PHER_MIN)
    norm = np.clip(ph / vmax, 0.0, 1.0)
    rgba = np.zeros((ph.shape[0], ph.shape[1], 4), dtype=float)
    rgba[..., 0] = 0.0   # green
    rgba[..., 1] = 1.0
    rgba[..., 2] = 0.3
    rgba[..., 3] = norm * alpha_scale
    return rgba

# -----------------------------
# Simulation step
# -----------------------------

def _maybe_trigger_failure():
    """Freeze the chosen robot exactly at FAIL_AT_STEP."""
    global failure_triggered
    if failure_triggered or FAIL_ROBOT_ID is None or FAIL_AT_STEP is None:
        return
    if global_step == FAIL_AT_STEP:
        robots[FAIL_ROBOT_ID].failed = True
        failure_triggered = True

def save_targets_over_time_plot(path: Path):
    """Save dotted line plot: time step vs total targets detected (cumulative)."""
    import matplotlib.pyplot as plt
    xs = np.arange(len(targets_found_over_time))
    ys = np.asarray(targets_found_over_time, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, linestyle=':', linewidth=2)  # dotted
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total targets detected")
    ax.set_title("Targets detected over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def sim_step():
    global red_pher, green_pher, global_step

    # NEW: trigger a failure exactly at the configured step
    _maybe_trigger_failure()

    # Evaporate both pheromones
    red_pher *= np.exp(-1.0 / TAU_DECAY)
    red_pher[red_pher < PHER_MIN] = 0.0
    green_pher *= np.exp(-1.0 / TAU_DECAY)
    green_pher[green_pher < PHER_MIN] = 0.0

    # Sense/cover/discover; deposit pheromones
    newly_found = set()
    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)  # private
        mark_visible_bool(covered_global, r.x, r.y)   # visualization union
        
        # Track newly discovered targets
        before = len(found_targets)
        discover_vn(r.x, r.y, targets, found_targets, W, H)
        if len(found_targets) > before:
            # NEW: Emit GREEN pheromone when target found!
            green_pher[r.y, r.x] += GREEN_PHER_DEPOSIT
            newly_found.add(r.id)
        
        # RED pheromone: emit at regular intervals for exploration
        if global_step % RED_EMIT_INTERVAL == 0:
            red_pher[r.y, r.x] += RED_PHER_DEPOSIT

    # Move robots (failed ones won't move because Robot.step guards it)
    for r in robots:
        r.step(red_pher, green_pher)

    # Post-move sensing & smaller pheromone deposit
    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_global, r.x, r.y)
        
        before = len(found_targets)
        discover_vn(r.x, r.y, targets, found_targets, W, H)
        if len(found_targets) > before:
            green_pher[r.y, r.x] += 0.5 * GREEN_PHER_DEPOSIT
        
        if global_step % RED_EMIT_INTERVAL == 0:
            red_pher[r.y, r.x] += 0.3 * RED_PHER_DEPOSIT

    # NEW: time bookkeeping + cumulative target logging
    global_step += 1
    targets_found_over_time.append(len(found_targets))
    if len(found_targets) >= len(targets):
        print(f"\n✓ All targets discovered at step {global_step}")
        import sys; sys.exit(0)


def _all_targets_found() -> bool:
    return len(found_targets) >= len(targets)

# -----------------------------
# Animation update
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    world_cov_img.set_data(coverage_to_image(covered_global))
    # Combine red and green pheromones for visualization
    combined_pher = pheromone_to_rgba(red_pher)
    green_overlay = green_pheromone_to_rgba(green_pher)
    # Blend: green overrides red where present
    combined_pher[..., :3] = combined_pher[..., :3] * (1 - green_overlay[..., 3:4]) + green_overlay[..., :3] * green_overlay[..., 3:4]
    combined_pher[..., 3] = np.maximum(combined_pher[..., 3], green_overlay[..., 3])
    world_pher_img.set_data(combined_pher)
    
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
    obs_pher_img.set_data(combined_pher)

    # NEW: failed robots in red + label suffix
    colors = ['red' if r.failed else 'k' for r in robots]
    robot_scat.set_facecolors(colors)
    robot_scat.set_edgecolors(colors)
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.x + 0.6, r.y + 0.6))
        robot_labels[i].set_text(f"R{r.id}" + (" (failed)" if r.failed else ""))
        robot_labels[i].set_color('red' if r.failed else 'k')

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
        "World – Dual Stigmergy (Red=Exploration, Green=Target)\n"
        f"Covered (union): {covered_global.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )

    # NEW: save dotted targets-over-time plot once, when all targets are found
    global plot_saved
    if (not plot_saved) and _all_targets_found():
        try:
            out_dir = OUTPUT_DIR
        except NameError:
            out_dir = Path("out")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(out_dir / "targets_over_time_stigmergy_dual.png")
        
        if FAIL_ROBOT_ID is not None:
            np.save('output_metrics/stigmergy_dual_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/stigmergy_dual_without_failure.npy', np.array(targets_found_over_time))
        
        plot_saved = True

    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat, obs_pher_img, und_plot, disc_plot, *robot_labels)


if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    GRID_SIZE = 200
    N_ROBOTS = 10
    N_TARGETS = 5
    RANDOM_SEED = 7
    STEPS_PER_FRAME = 10
    INTERVAL_MS = 50

    # for metric - number of targets detected over time steps
    targets_found_over_time = []   # cumulative #found after each sim_step
    plot_saved = False             # guard so we only write once
    OUTPUT_DIR = Path("output_frames/stigmergy_random_walk/")

    # Failure scenario (NEW)
    FAIL_ROBOT_ID = None   # e.g., 2
    FAIL_AT_STEP  = None   # e.g., 800
    global_step = 0
    failure_triggered = False

    # Stigmergy / pheromone - DUAL SYSTEM
    RED_PHER_DEPOSIT = 1.0      # exploration pheromone (repulsive)
    GREEN_PHER_DEPOSIT = 5.0    # target pheromone (attractive)
    TAU_DECAY = 60.0
    PHER_MIN = 1e-6
    BIAS_ALPHA_RED = 250        # avoid red pheromone strength
    BIAS_ALPHA_GREEN = 150      # attracted to green pheromone
    UNCOVERED_BONUS = 10.0      # slight bonus for unexplored
    RED_EMIT_INTERVAL = 5       # emit red every N steps
    COVER_RADIUS_FOR_MOVE = 5
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
    pts = rng.random((N_ROBOTS, 2)) * np.array([W, H])
    # Robots spawn  
    all_cells = [(x, y) for x in range(W) for y in range(H)]
    spawn_idx = rng.choice(len(all_cells), size=N_ROBOTS, replace=False)
    spawn_positions = [all_cells[i] for i in spawn_idx]
    # robots = [Robot(i, x, y, local_covered=np.zeros((H, W), dtype=bool)) for i, (x, y) in enumerate(spawn_positions)]
    print(pts[0,0], pts[0,1])
    robots = [Robot(i, int(pts[i, 0]), int(pts[i,1]), local_covered=np.zeros((H, W), dtype=bool)) for i in range(N_ROBOTS)]

    # Targets & state
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    found_targets: Set[Tuple[int,int]] = set()

    # Global (for visualization only — NOT shared by robots)
    covered_global = np.zeros((H, W), dtype=bool)
    red_pher = np.zeros((H, W), dtype=float)    # exploration pheromone
    green_pher = np.zeros((H, W), dtype=float)  # target pheromone
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
    world_pher_img = ax_world.imshow(pheromone_to_rgba(red_pher), origin='lower',
                                    extent=[0, W, 0, H], zorder=1)

    # Faint grid
    ax_world.set_xticks(np.arange(0, W+1, 10)); ax_world.set_yticks(np.arange(0, H+1, 10))
    ax_world.set_xticks(np.arange(0, W+1, 1), minor=True); ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    robot_colors = ['k' for _ in robots]  # black initially
    robot_scat = ax_world.scatter([r.x + 0.5 for r in robots],
                                [r.y + 0.5 for r in robots],
                                s=40, marker='o', c=robot_colors, zorder=3)

    # NEW: text labels above robots
    robot_labels = []
    for r in robots:
        t = ax_world.text(r.x + 0.6, r.y + 0.6, f"R{r.id}",
                        fontsize=7, color='k', zorder=5)
        robot_labels.append(t)

    # Targets
    if targets:
        tx, ty = zip(*targets)
    else:
        tx, ty = [], []
    ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)

    ax_world.set_title("World – Dual Stigmergy (Red=Exploration, Green=Target)")
    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')

    # Observer view (not shared by robots)
    obs_pher_img = ax_obs.imshow(pheromone_to_rgba(red_pher), origin='lower',
                                extent=[0, W, 0, H], zorder=0)
    und_plot = ax_obs.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                            s=18, marker='x', c='r', label='Undiscovered', zorder=2)
    disc_plot = ax_obs.scatter([], [], s=25, marker='o', facecolors='none',
                            edgecolors='g', linewidths=1.5, label='Discovered', zorder=2)

    ax_obs.set_xticks(np.arange(0, W+1, 10)); ax_obs.set_yticks(np.arange(0, H+1, 10))
    ax_obs.set_xticks(np.arange(0, W+1, 1), minor=True); ax_obs.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_obs.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_obs.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    ax_obs.set_title("Observer – Pheromone Fields (Red+Green overlay)")
    ax_obs.set_xlim(0, W); ax_obs.set_ylim(0, H); ax_obs.set_aspect('equal', adjustable='box')
    ax_obs.legend(loc='upper right', fontsize=8, frameon=False)

    

    # -----------------------------
    # Run
    # -----------------------------
    anim = run_animation(fig, update, frames=2000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()