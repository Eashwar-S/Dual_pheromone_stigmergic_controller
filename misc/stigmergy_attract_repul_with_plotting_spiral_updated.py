import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict, Deque
from collections import deque
from pathlib import Path
import random
from misc.simulation import FrameWriter, compute_fps, make_writer, run_animation

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
    
    # History to prevent loops (last 6 steps)
    follow_history: Deque[Tuple[int,int]] = field(default_factory=lambda: deque(maxlen=6))
    
    # Store utility terms for plotting (t1, t2, t3)
    current_terms: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Collision check helper
    def is_clear(self, nx: int, ny: int, all_robots: List['Robot'], radius: int = 1) -> bool:
        """
        Checks if moving to (nx, ny) violates the collision radius of any OTHER robot.
        Radius 1 means: Do not enter the 3x3 zone centered on another robot.
        """
        for r in all_robots:
            if r.id == self.id:
                continue
            # Chebyshev distance (square neighborhood) check
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    # ---------------------------------------------------------
    # 1. SEARCH MODE
    # ---------------------------------------------------------
    def calculate_utility(self, n_new_global, n_new_local, pher_sum):
        KAPPA = 61.0
        SIGMA = 10.0
        term1 = float(n_new_global)
        term2 = float(n_new_local) / KAPPA
        term3 = (np.tanh(-pher_sum / SIGMA) + 1.0) / (KAPPA ** 2)
        # Return both total score and the individual components
        return term1 + term2 + term3, (term1, term2, term3)

    def choose_move_search(self, pher_rep: np.ndarray, pher_attr: np.ndarray, all_robots: List['Robot']) -> Tuple[int, int]:
        H, W = pher_rep.shape
        
        # 1. Generate Valid Candidates (Collision & Bounds Checked)
        candidates = []
        # Check all 4 neighbors
        for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if self.is_clear(nx, ny, all_robots, COLLISION_RADIUS):
                    candidates.append((nx, ny))
        
        # If completely blocked by robots/walls, stay put (and metrics stay 0)
        if not candidates: 
            self.current_terms = (0.0, 0.0, 0.0)
            return (self.x, self.y)

        # 2. Check Active Escape Mode (Priority Over Utility)
        # This aligns with Pseudocode: "If k_esc > 0 ..."
        if hasattr(self, "_escape_steps") and self._escape_steps > 0:
            self._escape_steps -= 1
            # Pseudocode: "Random move from N biased by Phi"
            # Implementation: Random valid move is sufficient to break corner traps
            return random.choice(candidates)

        # 3. Compute Utilities
        R = 5
        KAPPA = 61.0
        SIGMA = 10.0
        scored_moves = []
        
        for (nx, ny) in candidates:
            # Lookahead: What will I see if I move to (nx, ny)?
            u1_val = 0      # Global Novelty
            u2_val = 0      # Local Novelty
            pher_sum = 0.0  # Pheromone Sum
            
            # Scan the sensor footprint of the CANDIDATE position
            for dy in range(-R, R+1):
                for dx in range(-R, R+1):
                    if abs(dx)+abs(dy) <= R:
                        cx, cy = nx+dx, ny+dy
                        if 0<=cx<W and 0<=cy<H:
                            # Cache values to avoid repeated array access
                            is_visited = self.local_covered[cy, cx]
                            p_val = pher_rep[cy, cx]
                            
                            pher_sum += p_val
                            
                            if not is_visited:
                                u2_val += 1
                                if p_val < 1e-3: # Approx 0 check
                                    u1_val += 1
            
            # Calculate Psi terms
            term1 = float(u1_val)
            term2 = float(u2_val) / KAPPA
            
            # u3 is negative sum (Repulsion)
            # Formula: (tanh(u3/sigma) + 1) / k^2
            # Since u3 is negative, we pass -pher_sum
            u3_input = -pher_sum
            term3 = (np.tanh(u3_input / SIGMA) + 1.0) / (KAPPA ** 2)
            
            psi = term1 + term2 + term3
            
            scored_moves.append({
                'move': (nx, ny),
                'score': psi,
                'terms': (term1, term2, term3),
                'u1_raw': u1_val,
                'u2_raw': u2_val
            })

        # 4. Selection
        best = max(scored_moves, key=lambda x: x['score'])
        self.current_terms = best['terms']

        # 5. Stagnation Check
        if best['u1_raw'] == 0 and best['u2_raw'] == 0:
            self._escape_steps = 5 # Set countdown (k const)
            
            # Trigger escape immediately by taking a random move NOW
            # This prevents wasting a step staying in the stuck spot
            return random.choice(candidates)
        
        return best['move']

    # ---------------------------------------------------------
    # 2. FOLLOW MODE (Vector Center-of-Gravity + Taboo)
    # ---------------------------------------------------------
    def choose_move_follow(self, pher_attr: np.ndarray, all_robots: List['Robot']) -> Tuple[int, int]:
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

            # 2. COLLISION CHECK: Don't move if blocked
            if not self.is_clear(nx, ny, all_robots, COLLISION_RADIUS):
                continue

            # 3. Alignment Score
            alignment = (dx * vec_x) + (dy * vec_y)
            intensity = pher_attr[ny, nx]
            score = alignment + (intensity * 0.1)

            if score > best_score:
                best_score = score
                best_move = (nx, ny)
        
        # If all moves are taboo/blocked
        if best_move is None:
            self.follow_history.clear()
            max_p = -1
            fallback = (self.x, self.y)
            for (dx, dy, nx, ny) in candidates:
                 if self.is_clear(nx, ny, all_robots, COLLISION_RADIUS): # Check safety
                     if pher_attr[ny, nx] > max_p:
                         max_p = pher_attr[ny, nx]
                         fallback = (nx, ny)
            return fallback

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
        signal_strength = 10.0/(0.1*dist_to_target + 0.00001)

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
    def step(self, pher_rep: np.ndarray, pher_attr: np.ndarray, targets: Set[Tuple[int,int]], all_robots: List['Robot']):
        H, W = pher_rep.shape
        
        # Reset utility terms to 0.0 if not in search (will be overwritten if in search)
        self.current_terms = (0.0, 0.0, 0.0)

        # A. TARGET SENSING & OVERRIDE
        if (self.x, self.y) in targets:
             if (self.x, self.y) not in self.visited_targets:
                self.visited_targets.add((self.x, self.y))
                if (self.x, self.y) not in global_target_visits:
                    global_target_visits[(self.x, self.y)] = set()
                global_target_visits[(self.x, self.y)].add(self.id)

                self.mode = "ADVERTISE"
                self.advertising_target = (self.x, self.y)
                # Initialize/Reset Spiral
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
                
                if 0 <= nx < W and 0 <= ny < H and self.is_clear(nx, ny, all_robots, COLLISION_RADIUS):
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
                # Use is_clear to ensure spiral doesn't crash into another robot
                if self.is_clear(nx, ny, all_robots, COLLISION_RADIUS):
                    self.x, self.y = nx, ny
            
            # Deposit signal at new location (creates the spiral trail)
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
                self.follow_history.clear() 
                nx, ny = self.choose_move_follow(pher_attr, all_robots)
                self.x, self.y = nx, ny
            else:
                nx, ny = self.choose_move_search(pher_rep, pher_attr, all_robots)
                self.x, self.y = nx, ny
                self.deposit_repulsive(pher_rep, amount=PHER_DEPOSIT, r=5)

        elif self.mode == "FOLLOW":
            nx, ny = self.choose_move_follow(pher_attr, all_robots)
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
    
    # MODIFIED: Attractive pheromone does NOT evaporate to maintain spiral trail
    # pher_attract *= decay 
    
    pher_repulse[pher_repulse < 1e-5] = 0.0
    pher_attract[pher_attract < 1e-5] = 0.0

    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_global, r.x, r.y)
        discover_vn(r.x, r.y, targets, found_targets, W, H)

    for r in robots:
        r.step(pher_repulse, pher_attract, targets, robots)
        
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

        # Update plotting data for utility terms
        for r_idx, r in enumerate(robots):
            if r_idx == 0:
                R0_data['t1'].append(r.current_terms[0])
                R0_data['t2'].append(r.current_terms[1])
                R0_data['t3'].append(r.current_terms[2])
            elif r_idx == 1:
                R1_data['t1'].append(r.current_terms[0])
                R1_data['t2'].append(r.current_terms[1])
                R1_data['t3'].append(r.current_terms[2])

    # 1. Update Existing Plots
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

    # 2. Update New Live Plots
    # Robot 0
    lines0[0].set_data(range(len(R0_data['t1'])), R0_data['t1'])
    lines0[1].set_data(range(len(R0_data['t2'])), R0_data['t2'])
    lines0[2].set_data(range(len(R0_data['t3'])), R0_data['t3'])
    ax_r0.relim(); ax_r0.autoscale_view()

    # Robot 1
    lines1[0].set_data(range(len(R1_data['t1'])), R1_data['t1'])
    lines1[1].set_data(range(len(R1_data['t2'])), R1_data['t2'])
    lines1[2].set_data(range(len(R1_data['t3'])), R1_data['t3'])
    ax_r1.relim(); ax_r1.autoscale_view()

    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat, *lines0, *lines1)

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
    COLLISION_RADIUS = 1 # 1 = 3x3 block safe
    
    OUTPUT_DIR = Path("output_frames/stigmergy_rendezvous/")
    metrics_log = [] 

    # Buffers for Live Plotting
    PLOT_LEN = 200
    R0_data = {'t1': deque(maxlen=PLOT_LEN), 't2': deque(maxlen=PLOT_LEN), 't3': deque(maxlen=PLOT_LEN)}
    R1_data = {'t1': deque(maxlen=PLOT_LEN), 't2': deque(maxlen=PLOT_LEN), 't3': deque(maxlen=PLOT_LEN)}

    PHER_DEPOSIT = 1.0
    TAU_DECAY = 200.0 
    rng = np.random.default_rng(RANDOM_SEED)

    FPS = compute_fps(INTERVAL_MS)
    frame_writer = FrameWriter(str(OUTPUT_DIR))

    W = H = GRID_SIZE
    global_step = 0
    
    pts = np.array([[10, 10], [90, 90]])
    
    robots = [Robot(
        i, 
        int(pts[i, 0]), 
        int(pts[i,1]), 
        local_covered=np.zeros((H, W), dtype=bool)
    ) for i in range(N_ROBOTS)]

    targets = {(50, 50)}
    found_targets = set()

    covered_global = np.zeros((H, W), dtype=bool)
    pher_repulse = np.zeros((H, W), dtype=float)
    pher_attract = np.zeros((H, W), dtype=float)

    # -----------------------------
    # Plotting Initialization
    # -----------------------------
    fig = plt.figure(figsize=(12, 10)) # Increased height for new plots
    gs = fig.add_gridspec(2, 2)
    
    # Top Row: Existing Plots
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs   = fig.add_subplot(gs[0, 1])

    # Bottom Row: New Utility Plots
    ax_r0 = fig.add_subplot(gs[1, 0])
    ax_r1 = fig.add_subplot(gs[1, 1])

    # -- World Map Setup --
    world_cov_img = ax_world.imshow(coverage_to_image(covered_global), origin='lower', extent=[0, W, 0, H], vmin=0, vmax=1, zorder=0)
    world_pher_img = ax_world.imshow(combined_pheromone_to_rgba(pher_repulse, pher_attract), origin='lower', extent=[0, W, 0, H], zorder=1)
    robot_scat = ax_world.scatter([], [], s=40, zorder=5)
    tx, ty = zip(*targets)
    ax_world.scatter([x+0.5 for x in tx], [y+0.5 for y in ty], s=40, marker='x', c='r', zorder=4)
    robot_labels = [ax_world.text(0,0,"", fontsize=7, color='white', fontweight='bold', zorder=6) for _ in robots]
    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H)
    ax_world.set_title("World Map (Black=Trail, Blue=Target)")

    # -- Pheromone Map Setup --
    obs_pher_img = ax_obs.imshow(combined_pheromone_to_rgba(pher_repulse, pher_attract), origin='lower', extent=[0, W, 0, H])
    disc_plot = ax_obs.scatter([], [], s=50, marker='o', edgecolors='g', facecolors='none')
    ax_obs.set_title("Pheromone Field (Detail)")

    # -- Robot 0 Utility Plot Setup --
    lines0 = []
    lines0.append(ax_r0.plot([], [], label='T1 (Glb Expl)', color='red')[0])
    lines0.append(ax_r0.plot([], [], label='T2 (Loc Expl)', color='green')[0])
    lines0.append(ax_r0.plot([], [], label='T3 (Pher Rep)', color='blue')[0])
    ax_r0.set_title("Robot 0 Utility Terms")
    ax_r0.legend(loc='upper left', fontsize='small')
    ax_r0.grid(True)

    # -- Robot 1 Utility Plot Setup --
    lines1 = []
    lines1.append(ax_r1.plot([], [], label='T1 (Glb Expl)', color='red')[0])
    lines1.append(ax_r1.plot([], [], label='T2 (Loc Expl)', color='green')[0])
    lines1.append(ax_r1.plot([], [], label='T3 (Pher Rep)', color='blue')[0])
    ax_r1.set_title("Robot 1 Utility Terms")
    ax_r1.legend(loc='upper left', fontsize='small')
    ax_r1.grid(True)

    plt.tight_layout()
    anim = run_animation(fig, update, frames=1500, interval_ms=INTERVAL_MS, blit=False)
    plt.show()