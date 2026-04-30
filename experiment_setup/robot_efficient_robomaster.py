import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
from pheromone import deposit_uniform


@dataclass
class Robot:
    """Robot using SEARCH mode with repulsive pheromone and utility-based movement."""
    id: int
    x: int
    y: int
    local_covered: np.ndarray
    failed: bool = False
    heading: int = 0  # 0: Up (-y), 90: Right (+x), 180: Down (+y), 270: Left (-x)

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

    def get_fov_cells(self, cx: int, cy: int, heading: int, radius: int, W: int, H: int) -> List[Tuple[int, int]]:
        """Return the coordinates of cells within the directional FOV."""
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    if heading == 0 and dy > 0: continue     # Up
                    if heading == 90 and dx < 0: continue    # Right
                    if heading == 180 and dy < 0: continue   # Down
                    if heading == 270 and dx > 0: continue   # Left
                    
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        cells.append((nx, ny))
        return cells

    def get_next_state(self, action: str) -> Tuple[int, int, int]:
        """Compute the resulting state (x, y, heading) for a given action."""
        nx, ny, nheading = self.x, self.y, self.heading
        
        # Get forward vector
        if self.heading == 0: fx, fy = 0, -1
        elif self.heading == 90: fx, fy = 1, 0
        elif self.heading == 180: fx, fy = 0, 1
        elif self.heading == 270: fx, fy = -1, 0
        else: fx, fy = 0, 0
        
        # Get right vector
        rx, ry = -fy, fx
        
        if action == 'FORWARD': nx, ny = self.x + fx, self.y + fy
        elif action == 'BACKWARD': nx, ny = self.x - fx, self.y - fy
        elif action == 'STRAFE_RIGHT': nx, ny = self.x + rx, self.y + ry
        elif action == 'STRAFE_LEFT': nx, ny = self.x - rx, self.y - ry
        elif action == 'ROTATE_RIGHT': nheading = (self.heading + 90) % 360
        elif action == 'ROTATE_LEFT': nheading = (self.heading - 90) % 360
        
        return nx, ny, nheading

    def choose_move_search(self, pher_rep: np.ndarray, all_robots: List['Robot'],
                       robot_radius: int, collision_radius: int = 1) -> str:

        H, W = pher_rep.shape
        actions = ['FORWARD', 'BACKWARD', 'STRAFE_LEFT', 'STRAFE_RIGHT', 'ROTATE_LEFT', 'ROTATE_RIGHT']
        scored_actions = []

        for action in actions:
            nx, ny, nheading = self.get_next_state(action)
            
            # If translation, check bounds and collision
            if action not in ['ROTATE_LEFT', 'ROTATE_RIGHT']:
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if not self.is_clear(nx, ny, all_robots, collision_radius):
                    continue
            
            # Compute metrics for the NEW FOV
            fov_cells = self.get_fov_cells(nx, ny, nheading, robot_radius, W, H)
            
            u1_val = 0
            u2_val = 0
            pher_sum = 0.0
            self_visited = 0

            for (cx, cy) in fov_cells:
                is_visited = self.local_covered[cy, cx]
                p_val = pher_rep[cy, cx]
                pher_sum += p_val

                if not is_visited:
                    u2_val += 1
                    if p_val == 0.0:
                        u1_val += 1
                else:
                    self_visited += 1

            scored_actions.append({
                'action': action,
                'u1': u1_val,
                'u2': u2_val,
                'pher': pher_sum,
                'self': self_visited
            })

        if not scored_actions:
            return 'STAY'

        # ============================
        # LEXICOGRAPHIC SELECTION
        # ============================

        # maximize u1
        max_u1 = max(m['u1'] for m in scored_actions)
        best_u1 = [m for m in scored_actions if m['u1'] == max_u1]

        # maximize u2 within u1-best
        max_u2 = max(m['u2'] for m in best_u1)
        best_u2 = [m for m in best_u1 if m['u2'] == max_u2]

        if max_u1 > 0:
            return random.choice(best_u2)['action']

        # Fallback (normalization approach)
        pher_vals = np.array([m['pher'] for m in scored_actions], dtype=float)
        self_vals = np.array([m['self'] for m in scored_actions], dtype=float)

        pher_range = np.ptp(pher_vals)  # max - min
        self_range = np.ptp(self_vals)
        # normalize to [0, 1]
        pher_norm = (pher_vals - pher_vals.min()) / (pher_range + 1e-6)
        self_norm = (self_vals - self_vals.min()) / (self_range + 1e-6)

        costs = np.maximum(pher_norm, self_norm)
        
        # Add a slight penalty to rotations if we are falling back
        # to avoid useless spinning in place
        for i, m in enumerate(scored_actions):
            if m['action'] in ['ROTATE_LEFT', 'ROTATE_RIGHT']:
                costs[i] += 0.05
                
        min_cost = min(costs)
        best = [m for m, c in zip(scored_actions, costs) if c == min_cost]

        return random.choice(best)['action']

    def step(self, pher_rep: np.ndarray, all_robots: List['Robot'], 
             robot_radius: int, collision_radius: int = 1):
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        
        action = self.choose_move_search(pher_rep, all_robots, robot_radius, collision_radius)
        if action != 'STAY':
            nx, ny, nheading = self.get_next_state(action)
            self.x, self.y, self.heading = nx, ny, nheading

    def deposit_pheromone(self, pher: np.ndarray, amount: float, r: int = 5):
        """Deposit repulsive pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, r)
