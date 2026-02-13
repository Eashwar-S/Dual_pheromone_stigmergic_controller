# --- ADDITIONS/CHANGES START HERE (visualization code below remains unchanged) ---

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict
from pathlib import Path
from simulation import FrameWriter, compute_fps, make_writer, run_animation

# -----------------------------
# Utilities (existing + new)
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

# ---- NEW: 4-neighborhood (no center) ----
def neighbors_vn_4(x: int, y: int, W: int, H: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if y - 1 >= 0: out.append((x, y - 1))
    if y + 1 < H:  out.append((x, y + 1))
    if x - 1 >= 0: out.append((x - 1, y))
    if x + 1 < W:  out.append((x + 1, y))
    return out

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int):
    """Mark current cell + VN neighbors True on the provided boolean grid."""
    H, W = grid_bool.shape
    grid_bool[y, x] = True
    if y-1 >= 0: grid_bool[y-1, x] = True
    if y+1 < H:  grid_bool[y+1, x] = True
    if x-1 >= 0: grid_bool[y, x-1] = True
    if x+1 < W:  grid_bool[y, x+1] = True

# ---- NEW: mark + report if anything was new this call ----
def mark_and_check_new(grid_bool: np.ndarray, x: int, y: int) -> bool:
    H, W = grid_bool.shape
    new_found = False
    if not grid_bool[y, x]:
        grid_bool[y, x] = True; new_found = True
    if y-1 >= 0 and not grid_bool[y-1, x]:
        grid_bool[y-1, x] = True; new_found = True
    if y+1 < H and not grid_bool[y+1, x]:
        grid_bool[y+1, x] = True; new_found = True
    if x-1 >= 0 and not grid_bool[y, x-1]:
        grid_bool[y, x-1] = True; new_found = True
    if x+1 < W and not grid_bool[y, x+1]:
        grid_bool[y, x+1] = True; new_found = True
    return new_found

def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_vn(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

# ---- NEW: skip deposit if all 4 neighbors already have pheromone ----
def surrounded_by_pheromone(x: int, y: int, pher: np.ndarray) -> bool:
    H, W = pher.shape
    for nx, ny in neighbors_vn_4(x, y, W, H):
        if pher[ny, nx] <= 0.0:
            return False
    return True

# -----------------------------
# Robot with PRIVATE local map  (extended)
# -----------------------------
@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray  # (H,W) bool, private per-robot map
    last_move: Optional[Tuple[int,int]] = None
    failed: bool = False  # freezes in place once failed
    # ---- NEW anti-stagnation state ----
    stagnation_steps: int = 0
    random_walk_remaining: int = 0

    # ---- UPDATED: supports uniform (pheromone-agnostic) random walk ----
    def choose_move(self, pher: np.ndarray, uniform_random: bool = False) -> Tuple[int,int]:
        H, W = pher.shape
        candidates: List[Tuple[int,int]] = []
        if self.y-1 >= 0: candidates.append((self.x, self.y-1))
        if self.y+1 < H:  candidates.append((self.x, self.y+1))
        if self.x-1 >= 0: candidates.append((self.x-1, self.y))
        if self.x+1 < W:  candidates.append((self.x+1, self.y))
        if not candidates:
            return (self.x, self.y)

        if uniform_random:
            idx = rng.integers(len(candidates))
            return candidates[int(idx)]

        # Prefer locally-uncovered, avoid pheromone
        uncov = [(nx, ny) for (nx, ny) in candidates if not self.local_covered[ny, nx]]
        pool = uncov if len(uncov) > 0 else candidates
        weights = []
        for (nx, ny) in pool:
            p = max(pher[ny, nx], 0.0)
            desirability = np.exp(-BIAS_ALPHA * (p / (1.0 + p)))
            if not self.local_covered[ny, nx]:
                desirability *= UNCOVERED_BONUS
            weights.append(desirability)
        w = np.array(weights, dtype=float)
        if np.all(w <= 0):
            w = np.ones_like(w)
        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[idx]

    def step(self, pher: np.ndarray):
        if self.failed:
            return
        uniform = (self.random_walk_remaining > 0)
        nx, ny = self.choose_move(pher, uniform_random=uniform)
        self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        if self.random_walk_remaining > 0:
            self.random_walk_remaining -= 1

# -----------------------------
# Visualization helpers (unchanged)
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
# Multiple-failure support (NEW)
# -----------------------------
def _maybe_trigger_failures():
    """Freeze all robots whose fail step equals the current global_step."""
    if not FAILURE_EVENTS_BY_STEP:
        return
    if global_step in FAILURE_EVENTS_BY_STEP:
        for rid in FAILURE_EVENTS_BY_STEP[global_step]:
            if 0 <= rid < len(robots):
                robots[rid].failed = True

# -----------------------------
# Simulation step (UPDATED)
# -----------------------------
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
    global pher, global_step

    # NEW: trigger all failures scheduled at this step
    _maybe_trigger_failures()

    # ---- DISABLE EVAPORATION ----
    # (no decay applied)
    pher[pher < PHER_MIN] = 0.0

    # Pre-move sense/cover/discover + gated deposit
    any_new_step = [False] * len(robots)
    for r in robots:
        # mark local + global and detect if new ground was covered
        new_local = mark_and_check_new(r.local_covered, r.x, r.y)
        new_global = mark_and_check_new(covered_global, r.x, r.y)
        any_new = (new_local or new_global)
        any_new_step[r.id] = any_new
        discover_vn(r.x, r.y, targets, found_targets, W, H)

        # skip deposit if surrounded
        if not surrounded_by_pheromone(r.x, r.y, pher):
            pher[r.y, r.x] += PHER_DEPOSIT

    # Anti-stagnation: start random walk if needed
    for r in robots:
        if r.failed:
            continue
        if any_new_step[r.id]:
            r.stagnation_steps = 0
        else:
            r.stagnation_steps += 1
            if r.stagnation_steps >= STAGNATION_X and r.random_walk_remaining == 0:
                r.random_walk_remaining = RANDOMWALK_Y
                r.stagnation_steps = 0

    # Move robots (failed ones won't move because Robot.step guards it)
    for r in robots:
        r.step(pher)

    # Post-move sensing + gated small deposit
    for r in robots:
        new_local = mark_and_check_new(r.local_covered, r.x, r.y)
        new_global = mark_and_check_new(covered_global, r.x, r.y)
        if new_local or new_global:
            r.stagnation_steps = 0
        discover_vn(r.x, r.y, targets, found_targets, W, H)
        if not surrounded_by_pheromone(r.x, r.y, pher):
            pher[r.y, r.x] += 0.3 * PHER_DEPOSIT

    # time bookkeeping + cumulative target logging
    global_step += 1
    targets_found_over_time.append(len(found_targets))

def _all_targets_found() -> bool:
    return len(found_targets) >= len(targets)

# -----------------------------
# Animation update (visualization left intact)
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    world_cov_img.set_data(coverage_to_image(covered_global))
    world_pher_img.set_data(pheromone_to_rgba(pher))
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
    obs_pher_img.set_data(pheromone_to_rgba(pher))

    # failed robots in red + label suffix
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
        "World — Stigmergy (Local Maps + Random Walk + Pheromone)\n"
        f"Covered (union): {covered_global.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )

    # save dotted targets-over-time plot once, when all targets are found
    global plot_saved
    if (not plot_saved) and _all_targets_found():
        try:
            out_dir = OUTPUT_DIR
        except NameError:
            out_dir = Path("out")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(out_dir / "targets_over_time_stigmergy.png")
        
        # keep existing filenames
        if FAIL_EVENTS:
            np.save('output_metrics/stigmergy_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/stigmergy_without_failure.npy', np.array(targets_found_over_time))
        
        plot_saved = True

    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat, obs_pher_img, und_plot, disc_plot, *robot_labels)



# -----------------------------
# MAIN (visualization block below is UNCHANGED except globals)
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    GRID_SIZE = 25
    N_ROBOTS = 5
    N_TARGETS = 80
    RANDOM_SEED = 73550
    STEPS_PER_FRAME = 1
    INTERVAL_MS = 80

    # for metric - number of targets detected over time steps
    targets_found_over_time = []   # cumulative #found after each sim_step
    plot_saved = False             # guard so we only write once
    OUTPUT_DIR = Path("output_frames/stigmergy_random_walk/")

    # ---- MULTIPLE FAILURES (edit this list) ----
    # e.g., [(2, 100), (4, 250)]
    FAIL_EVENTS: List[Tuple[int,int]] = [(0, 47)]
    FAILURE_EVENTS_BY_STEP: Dict[int, List[int]] = {}
    for rid, st in FAIL_EVENTS:
        FAILURE_EVENTS_BY_STEP.setdefault(int(st), []).append(int(rid))

    global_step = 0

    # Stigmergy / pheromone
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 600.0          # kept for compatibility; NOT used (no evaporation)
    PHER_MIN = 1e-6
    BIAS_ALPHA = 1             # avoid-pheromone strength
    UNCOVERED_BONUS = 10.0
    # ---- NEW anti-stagnation knobs (fixed across a sweep/experiment) ----
    STAGNATION_X = 15          # x consecutive steps w/ no new ground
    RANDOMWALK_Y = 10          # then y uniform random-walk steps

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

    # Spawn all robots at center (as in your current script)
    robots = [Robot(i, robot_starting_x, robot_starting_y,
                    local_covered=np.zeros((H, W), dtype=bool)) for i in range(N_ROBOTS)]

    # Targets & state
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    found_targets: Set[Tuple[int,int]] = set()

    # Global (for visualization only — NOT shared by robots)
    covered_global = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)

    # -----------------------------
    # Matplotlib layout (UNTOUCHED)
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

    robot_colors = ['k' for _ in robots]  # black initially
    robot_scat = ax_world.scatter([r.x + 0.5 for r in robots],
                                [r.y + 0.5 for r in robots],
                                s=40, marker='o', c=robot_colors, zorder=3)

    # text labels above robots
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
# --- END OF FILE ---
