import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict, Deque
from collections import deque
from pathlib import Path
import random
from simulation import FrameWriter, compute_fps, make_writer, run_animation

# -----------------------------
# Utilities
# -----------------------------
def neighbors_vn_r(x: int, y: int, W: int, H: int, r: int = 5):
    out = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:
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

def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_vn_r(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

def sparse_spiral_generator(gap: int = 2):
    """
    Yields (dx, dy) for a sparse square spiral.
    Gap=2 means 2 empty cells between arms.
    """
    step_length = 1
    increment = gap + 1
    direction = 0  # 0=R, 1=D, 2=L, 3=U
    moves = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    
    while True:
        for _ in range(2):
            dx, dy = moves[direction]
            for _ in range(step_length):
                yield (dx, dy)
            direction = (direction + 1) % 4
        step_length += increment

# -----------------------------
# Robot Logic
# -----------------------------
@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray
    mode: str = "SEARCH"
    spiral_iter: Optional[object] = None
    visited_targets: Set[Tuple[int,int]] = field(default_factory=set)
    advertising_target: Optional[Tuple[int,int]] = None
    
    # NEW: History to prevent loops (last 6 steps)
    follow_history: Deque[Tuple[int,int]] = field(default_factory=lambda: deque(maxlen=6))
    
    # ---------------------------------------------------------
    # 1. SEARCH MODE
    # ---------------------------------------------------------
    def calculate_utility(self, n_new_global, n_new_local, pher_sum):
        KAPPA = 200.0
        SIGMA = 10.0
        term1 = float(n_new_global)
        term2 = float(n_new_local) / KAPPA
        term3 = (np.tanh(-pher_sum / SIGMA) + 1.0) / (KAPPA ** 2)
        return term1 + term2 + term3

    def choose_move_search(self, pher_rep: np.ndarray, pher_attr: np.ndarray) -> Tuple[int, int]:
        H, W = pher_rep.shape
        candidates = []
        if self.y - 1 >= 0: candidates.append((self.x, self.y - 1))
        if self.y + 1 < H:  candidates.append((self.x, self.y + 1))
        if self.x - 1 >= 0: candidates.append((self.x - 1, self.y))
        if self.x + 1 < W:  candidates.append((self.x + 1, self.y))
        
        random.shuffle(candidates)
        if not candidates: return (self.x, self.y)

        R = 5
        scored_moves = []
        for (nx, ny) in candidates:
            nb_cells = []
            for dy in range(-R, R+1):
                for dx in range(-R, R+1):
                    if abs(dx)+abs(dy) <= R:
                        cx, cy = nx+dx, ny+dy
                        if 0<=cx<W and 0<=cy<H:
                            nb_cells.append((cx, cy))
            
            new_anyone = sum(1 for (cx, cy) in nb_cells if (not self.local_covered[cy, cx]) and (pher_rep[cy, cx] < 1e-3))
            new_me     = sum(1 for (cx, cy) in nb_cells if not self.local_covered[cy, cx])
            pher_sum   = float(np.sum([pher_rep[cy, cx] for (cx, cy) in nb_cells]))
            
            score = self.calculate_utility(new_anyone, new_me, pher_sum)
            scored_moves.append({'move':(nx,ny), 'score':score, 'u1':new_anyone, 'pher':pher_rep[ny,nx]})

        best = max(scored_moves, key=lambda x: x['score'])

        # Escape Logic
        if best['u1'] == 0:
            if not hasattr(self, "_escape_steps") or self._escape_steps <= 0:
                min_edge = min(m['pher'] for m in scored_moves)
                starters = [m['move'] for m in scored_moves if m['pher'] == min_edge]
                ex, ey = random.choice(starters)
                self._escape_dir = (ex - self.x, ey - self.y)
                self._escape_steps = 15
            
            dx, dy = getattr(self, "_escape_dir", (0, 0))
            tx, ty = self.x + dx, self.y + dy
            if 0 <= tx < W and 0 <= ty < H:
                self._escape_steps -= 1
                return (tx, ty)
            else:
                self._escape_steps = 0
                return random.choice(candidates)
        
        if hasattr(self, "_escape_steps"): self._escape_steps = 0
        return best['move']

    # ---------------------------------------------------------
    # 2. FOLLOW MODE (Vector Center-of-Gravity + Taboo)
    # ---------------------------------------------------------
    def choose_move_follow(self, pher_attr: np.ndarray) -> Tuple[int, int]:
        H, W = pher_attr.shape
        R = 5  # Sensor radius
        
        # A. Calculate "Center of Gravity" of visible blue pheromone
        vec_x, vec_y = 0.0, 0.0
        total_weight = 0.0
        
        found_blue = False
        
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if abs(dx) + abs(dy) <= R:
                    cx, cy = self.x + dx, self.y + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        val = pher_attr[cy, cx]
                        if val > 1e-6:
                            found_blue = True
                            # Vector points from Robot -> Pheromone Cell
                            # Weight by intensity squared to bias towards bright centers
                            weight = val * val 
                            vec_x += dx * weight
                            vec_y += dy * weight
                            total_weight += weight

        if not found_blue:
            # Lost trail completely?
            self.mode = "SEARCH"
            return (self.x, self.y)

        # Normalize vector
        if total_weight > 0:
            length = np.sqrt(vec_x**2 + vec_y**2)
            if length > 0:
                vec_x /= length
                vec_y /= length
        else:
            # Should not happen if found_blue is True
            vec_x, vec_y = 0, 0

        # B. Score neighbors based on alignment with Vector
        candidates = []
        if self.y - 1 >= 0: candidates.append((0, -1, self.x, self.y - 1)) # dx, dy, x, y
        if self.y + 1 < H:  candidates.append((0, 1, self.x, self.y + 1))
        if self.x - 1 >= 0: candidates.append((-1, 0, self.x - 1, self.y))
        if self.x + 1 < W:  candidates.append((1, 0, self.x + 1, self.y))
        
        random.shuffle(candidates)

        best_move = None
        best_score = -999.0
        
        for (dx, dy, nx, ny) in candidates:
            # 1. TABOO CHECK: Don't go back to recently visited cells
            if (nx, ny) in self.follow_history:
                continue

            # 2. Alignment Score (Dot product)
            # How well does this move align with the Pheromone Vector?
            alignment = (dx * vec_x) + (dy * vec_y)
            
            # 3. Intensity Bonus (Greedy secondary check)
            # Slightly prefer cells that are brighter themselves
            intensity = pher_attr[ny, nx]
            
            score = alignment + (intensity * 0.1)

            if score > best_score:
                best_score = score
                best_move = (nx, ny)
        
        # If all moves are taboo (stuck in a corner/loop), reset history and pick random valid
        if best_move is None:
            self.follow_history.clear()
            # Just pick the brightest neighbor as fallback
            max_p = -1
            fallback = (self.x, self.y)
            for (dx, dy, nx, ny) in candidates:
                 if pher_attr[ny, nx] > max_p:
                     max_p = pher_attr[ny, nx]
                     fallback = (nx, ny)
            return fallback

        # Log history
        self.follow_history.append(best_move)
        return best_move

    # ---------------------------------------------------------
    # 3. DEPOSITION (Exponential Peak)
    # ---------------------------------------------------------
    def deposit_distance_signal(self, pher: np.ndarray, r: int = 5):
        if self.advertising_target is None: return

        tx, ty = self.advertising_target
        H, W = pher.shape
        x0, y0 = int(self.x), int(self.y)
        
        dist_to_target = np.sqrt((x0 - tx)**2 + (y0 - ty)**2)
        
        # EXPONENTIAL DECAY: Creates a very sharp peak at the center
        # At 0 dist = 10.0
        # At 20 dist = 10 * e^(-2) = 1.35
        # Much harder to miss the "uphill" direction
        signal_strength = 10.0/(0.1*dist_to_target + 0.00001)#np.exp(-0.01 * dist_to_target)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    cx, cy = x0 + dx, y0 + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        pher[cy, cx] = np.maximum(pher[cy, cx], signal_strength)

    def deposit_repulsive(self, pher: np.ndarray, amount, r):
        H, W = pher.shape
        x0, y0 = int(self.x), int(self.y)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    cx, cy = x0 + dx, y0 + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        pher[cy, cx] += amount

    # ---------------------------------------------------------
    # 4. STEP
    # ---------------------------------------------------------
    def step(self, pher_rep: np.ndarray, pher_attr: np.ndarray, targets: Set[Tuple[int,int]]):
        H, W = pher_rep.shape

        # A. TARGET SENSING & OVERRIDE
        if (self.x, self.y) in targets:
             if (self.x, self.y) not in self.visited_targets:
                self.visited_targets.add((self.x, self.y))
                if (self.x, self.y) not in global_target_visits:
                    global_target_visits[(self.x, self.y)] = set()
                global_target_visits[(self.x, self.y)].add(self.id)

                self.mode = "ADVERTISE"
                self.advertising_target = (self.x, self.y)
                # Use Gap=2 for 2 empty cells between arms
                self.spiral_iter = sparse_spiral_generator(gap=4)
                self.deposit_distance_signal(pher_attr, r=5)
                print(f"Robot {self.id} ACTIVATED target at {(self.x, self.y)}")

        if self.mode != "ADVERTISE":
            nearby_targets = []
            R = 5
            for t in targets:
                dist = abs(t[0] - self.x) + abs(t[1] - self.y)
                if dist <= R:
                    nearby_targets.append((t, dist))
            
            if nearby_targets:
                nearby_targets.sort(key=lambda x: x[1])
                target_pos = nearby_targets[0][0]
                dx = np.sign(target_pos[0] - self.x)
                dy = np.sign(target_pos[1] - self.y)
                if dx != 0 and dy != 0:
                    if random.random() < 0.5: dx = 0
                    else: dy = 0
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    self.x, self.y = nx, ny
                return

        # B. EXECUTE MODE
        if self.mode == "ADVERTISE":
            if self.spiral_iter is None: self.spiral_iter = sparse_spiral_generator(gap=4)
            try:
                dx, dy = next(self.spiral_iter)
            except:
                self.spiral_iter = sparse_spiral_generator(gap=4)
                dx, dy = next(self.spiral_iter)
            
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < W and 0 <= ny < H:
                self.x, self.y = nx, ny
            
            self.deposit_distance_signal(pher_attr, r=5)
        
        elif self.mode == "SEARCH":
            found_trail = False
            R = 5
            for dy in range(-R, R+1):
                for dx in range(-R, R+1):
                    if abs(dx)+abs(dy) <= R:
                        cx, cy = self.x+dx, self.y+dy
                        if 0<=cx<W and 0<=cy<H:
                            if pher_attr[cy, cx] > 1e-5:
                                found_trail = True
                                break
                if found_trail: break

            if found_trail:
                self.mode = "FOLLOW"
                # Clear history when entering follow mode to start fresh
                self.follow_history.clear() 
                nx, ny = self.choose_move_follow(pher_attr)
                self.x, self.y = nx, ny
            else:
                nx, ny = self.choose_move_search(pher_rep, pher_attr)
                self.x, self.y = nx, ny
                self.deposit_repulsive(pher_rep, amount=PHER_DEPOSIT, r=5)

        elif self.mode == "FOLLOW":
            nx, ny = self.choose_move_follow(pher_attr)
            
            if nx == self.x and ny == self.y:
                if (nx, ny) not in targets:
                    self.mode = "SEARCH"
            
            self.x, self.y = nx, ny

        # C. POST-MOVE CHECKS
        if (self.x, self.y) in targets:
            if (self.x, self.y) not in self.visited_targets:
                self.visited_targets.add((self.x, self.y))
                if (self.x, self.y) not in global_target_visits:
                    global_target_visits[(self.x, self.y)] = set()
                global_target_visits[(self.x, self.y)].add(self.id)
                
                self.mode = "ADVERTISE"
                self.advertising_target = (self.x, self.y)
                self.spiral_iter = sparse_spiral_generator(gap=4)
                self.deposit_distance_signal(pher_attr, r=5)
                print(f"Robot {self.id} found target at {(self.x, self.y)}")

# -----------------------------
# Visualization
# -----------------------------
def coverage_to_image(cv: np.ndarray) -> np.ndarray:
    img = np.ones(cv.shape, dtype=float)
    img[cv] = 0.85
    return img

def combined_pheromone_to_rgba(p_rep: np.ndarray, p_attr: np.ndarray) -> np.ndarray:
    H, W = p_rep.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    
    max_rep = np.percentile(p_rep, 98) if np.max(p_rep) > 0 else 1.0
    norm_rep = np.clip(p_rep / (max_rep + 1e-9), 0, 1) * 0.5
    
    # Scale Blue relative to max distance signal (10.0)
    norm_attr = np.clip(p_attr / 10.0, 0, 1) * 0.9
    
    rgba[..., 0] = norm_rep * 1.0 
    rgba[..., 1] = norm_rep * 0.2 + norm_attr * 0.2
    rgba[..., 2] = norm_rep * 0.6 + norm_attr * 1.0
    rgba[..., 3] = np.clip(norm_rep + norm_attr, 0, 1)
    
    return np.clip(rgba, 0.0, 1.0)

# -----------------------------
# Simulation State
# -----------------------------
global_target_visits: Dict[Tuple[int,int], Set[int]] = {}

def sim_step():
    global pher_repulse, pher_attract, global_step

    decay = np.exp(-1.0 / TAU_DECAY)
    pher_repulse *= decay
    pher_attract *= decay
    pher_repulse[pher_repulse < 1e-5] = 0.0
    pher_attract[pher_attract < 1e-5] = 0.0

    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_global, r.x, r.y)
        discover_vn(r.x, r.y, targets, found_targets, W, H)

    for r in robots:
        r.step(pher_repulse, pher_attract, targets)
        
    all_targets_satisfied = False
    if len(targets) > 0:
        met_targets = 0
        for t in targets:
            visitors = global_target_visits.get(t, set())
            if len(visitors) > 1:
                met_targets += 1
        if met_targets == len(targets):
            all_targets_satisfied = True

    global_step += 1
    
    targets_satisfied_count = sum(1 for t in targets if len(global_target_visits.get(t, set())) > 1)
    metrics_log.append((len(found_targets), targets_satisfied_count))

    if all_targets_satisfied:
        print(f"\n SUCCESS: All targets visited by >1 robot at step {global_step}")
        import sys; sys.exit(0)

# -----------------------------
# Animation Update
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    world_cov_img.set_data(coverage_to_image(covered_global))
    combined_rgba = combined_pheromone_to_rgba(pher_repulse, pher_attract)
    world_pher_img.set_data(combined_rgba)
    obs_pher_img.set_data(combined_rgba)

    offsets = np.array([[r.x + 0.5, r.y + 0.5] for r in robots])
    robot_scat.set_offsets(offsets)
    
    colors = []
    for r in robots:
        if r.mode == "SEARCH": colors.append('k')
        elif r.mode == "ADVERTISE": colors.append('blue') 
        elif r.mode == "FOLLOW": colors.append('purple')
    
    robot_scat.set_facecolors(colors)
    robot_scat.set_edgecolors(colors)

    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.x + 0.6, r.y + 0.6))
        robot_labels[i].set_text(f"R{r.id}")

    if targets:
        tx, ty = zip(*targets)
        t_colors = []
        for t in targets:
            visitors = global_target_visits.get(t, set())
            if len(visitors) > 1: t_colors.append('blue')
            elif len(visitors) == 1: t_colors.append('green')
            else: t_colors.append('red')
            
        ax_world.collections[1].remove()
        ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                         s=40, marker='x', c=t_colors, linewidths=2, zorder=4)

    discovered = list(found_targets)
    if discovered:
        dx, dy = zip(*discovered)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    
    title = f"Step {global_step}: Targets Satisfied (>1 visitor): {metrics_log[-1][1]}/{len(targets)}"
    ax_world.set_title(title)

    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat)

# -----------------------------
# Main Setup
# -----------------------------
if __name__ == "__main__":
    GRID_SIZE = 100 
    N_TARGETS = 1
    N_ROBOTS = 2    
    
    RANDOM_SEED = 42
    STEPS_PER_FRAME = 5
    INTERVAL_MS = 50
    
    OUTPUT_DIR = Path("output_frames/stigmergy_rendezvous/")
    metrics_log = [] 

    PHER_DEPOSIT = 1.0
    TAU_DECAY = 200.0 
    rng = np.random.default_rng(RANDOM_SEED)

    FPS = compute_fps(INTERVAL_MS)
    frame_writer = FrameWriter(str(OUTPUT_DIR))

    W = H = GRID_SIZE
    global_step = 0
    
    # pts = rng.random((N_ROBOTS, 2)) * np.array([W, H])
    # print(pts)
    pts = np.array([[10, 10], [90, 90]])
    robots = [Robot(i, int(pts[i, 0]), int(pts[i,1]), local_covered=np.zeros((H, W), dtype=bool)) for i in range(N_ROBOTS)]

    targets = {(50, 50)}
    found_targets = set()

    covered_global = np.zeros((H, W), dtype=bool)
    pher_repulse = np.zeros((H, W), dtype=float)
    pher_attract = np.zeros((H, W), dtype=float)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs   = fig.add_subplot(gs[0, 1])

    world_cov_img = ax_world.imshow(coverage_to_image(covered_global), origin='lower', extent=[0, W, 0, H], vmin=0, vmax=1, zorder=0)
    world_pher_img = ax_world.imshow(combined_pheromone_to_rgba(pher_repulse, pher_attract), origin='lower', extent=[0, W, 0, H], zorder=1)
    
    robot_scat = ax_world.scatter([], [], s=40, zorder=5)
    tx, ty = zip(*targets)
    ax_world.scatter([x+0.5 for x in tx], [y+0.5 for y in ty], s=40, marker='x', c='r', zorder=4)

    robot_labels = [ax_world.text(0,0,"", fontsize=7, color='white', fontweight='bold', zorder=6) for _ in robots]

    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H)
    
    obs_pher_img = ax_obs.imshow(combined_pheromone_to_rgba(pher_repulse, pher_attract), origin='lower', extent=[0, W, 0, H])
    disc_plot = ax_obs.scatter([], [], s=50, marker='o', edgecolors='g', facecolors='none')
    
    ax_obs.set_title("Pheromone Field (Bright Blue = Target Source)")

    anim = run_animation(fig, update, frames=1500, interval_ms=INTERVAL_MS, blit=False)
    plt.show()