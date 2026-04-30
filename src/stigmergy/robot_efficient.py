import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random
from .pheromone import deposit_uniform


@dataclass
class Robot:
    """Robot using SEARCH mode with repulsive pheromone and utility-based movement."""
    id: int
    x: int
    y: int
    local_covered: np.ndarray
    failed: bool = False
    _escape_steps: int = 0

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

    def choose_move_search(self, pher_rep: np.ndarray, all_robots: List['Robot'],
                       robot_radius: int, collision_radius: int = 1) -> Tuple[int, int]:

        H, W = pher_rep.shape

        # 1. Generate valid candidates
        candidates = []
        for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if self.is_clear(nx, ny, all_robots, collision_radius):
                    candidates.append((nx, ny))

        if not candidates:
            return (self.x, self.y)

        # 2. Escape mode (optional, keep if you want)
        if self._escape_steps > 0:
            self._escape_steps -= 1
            return random.choice(candidates)

        # 3. Compute metrics
        scored_moves = []

        for (nx, ny) in candidates:
            u1_val = 0
            u2_val = 0
            pher_sum = 0.0
            self_visited = 0

            for dy in range(-robot_radius, robot_radius + 1):
                for dx in range(-robot_radius, robot_radius + 1):
                    if abs(dx) + abs(dy) <= robot_radius:
                        cx, cy = nx + dx, ny + dy
                        if 0 <= cx < W and 0 <= cy < H:
                            is_visited = self.local_covered[cy, cx]
                            p_val = pher_rep[cy, cx]

                            pher_sum += p_val

                            if not is_visited:
                                u2_val += 1
                                if p_val == 0.0:
                                    u1_val += 1
                            else:
                                self_visited += 1

            scored_moves.append({
                'move': (nx, ny),
                'u1': u1_val,
                'u2': u2_val,
                'pher': pher_sum,
                'self': self_visited
            })

        # ============================
        # LEXICOGRAPHIC SELECTION
        # ============================

        # maximize u1
        max_u1 = max(m['u1'] for m in scored_moves)
        best_u1 = [m for m in scored_moves if m['u1'] == max_u1]

        # maximize u2 within u1-best
        max_u2 = max(m['u2'] for m in best_u1)
        best_u2 = [m for m in best_u1 if m['u2'] == max_u2]

        # if meaningful novelty exists → choose randomly among best
        # if max_u1 > 0 or max_u2 > 0:
        #     return random.choice(best_u2)['move']

        # If without pheromone is not present then randomly choose best unvisited area
        print("len of best_u2 =============================", len(best_u2))
        if max_u1 > 0:
            return random.choice(best_u2)['move']

        '''
        # If without pheromone is present then randomly choose best unvisited area
        # elif max_u2 > 0:
        #     return random.choice(best_u2)['move']


        # fallback → pheromone-weighted stochastic move
        # Lower pheromone = higher probability
        # pher_values = np.array([m['pher'] for m in scored_moves], dtype=float)

        # pher_values = pher_values - np.min(pher_values)

        # weights = np.exp(-pher_values)

        # total = weights.sum()

        # if total <= 1e-12 or not np.isfinite(total):
        #     weights = np.ones_like(weights) / len(weights)
        # else:
        #     weights /= total

        # idx = np.random.choice(len(scored_moves), p=weights)
        # return scored_moves[idx]['move']
        ''' # version 1

        # version 2 (epsilon-greedy) depends on epsilon
        # epsilon = 0.1  # small randomness

        # if np.random.rand() < epsilon:
        #     # exploration
        #     return random.choice(candidates)
        # else:
        #     # exploitation → pick lowest pheromone
        #     min_pher = min(m['pher'] for m in scored_moves)

        #     # handle ties (important!)
        #     best = [m for m in scored_moves if m['pher'] == min_pher]

        #     return random.choice(best)['move']

        # version 3 (pheromone-weighted stochastic move)
        # pher_values = np.array([m['pher'] for m in scored_moves], dtype=float)

        # pher_values = pher_values - np.min(pher_values)
        # scale = np.std(pher_values) + 1e-6
        # weights = np.exp(-(pher_values / scale))
        # print("weights =========================", weights)
        # total = weights.sum()

        # if total <= 1e-12 or not np.isfinite(total):
        #     weights = np.ones_like(weights) / len(weights)
        # else:
        #     weights /= total

        # idx = np.random.choice(len(scored_moves), p=weights)
        # return scored_moves[idx]['move']

        # version 4
        
        pher_vals = np.array([m['pher'] for m in scored_moves], dtype=float)
        self_vals = np.array([m['self'] for m in scored_moves], dtype=float)

        pher_range = np.ptp(pher_vals)  # max - min
        self_range = np.ptp(self_vals)
        # normalize to [0, 1]
        pher_norm = (pher_vals - pher_vals.min()) / (pher_range + 1e-6)
        self_norm = (self_vals - self_vals.min()) / (self_range + 1e-6)

        costs = np.maximum(pher_norm, self_norm)
        # costs = [m['pher'] + m['self'] for m in scored_moves]
        print("costs ========================", costs)
        print()
        min_cost = min(costs)
        best = [m for m, c in zip(scored_moves, costs) if c == min_cost]

        return random.choice(best)['move']

    def step(self, pher_rep: np.ndarray, all_robots: List['Robot'], 
             robot_radius: int, collision_radius: int = 1):
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        
        nx, ny = self.choose_move_search(pher_rep, all_robots, robot_radius, collision_radius)
        self.x, self.y = nx, ny

    def deposit_pheromone(self, pher: np.ndarray, amount: float, r: int = 5):
        """Deposit repulsive pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, r)



# This implementation is still in-efficient because of the utitlity function.
# @dataclass
# class Robot:
#     """Robot using SEARCH mode with repulsive pheromone and utility-based movement."""
#     id: int
#     x: int
#     y: int
#     local_covered: np.ndarray
#     failed: bool = False
#     _escape_steps: int = 0

#     def is_clear(self, nx: int, ny: int, all_robots: List['Robot'], radius: int = 1) -> bool:
#         """
#         Checks if moving to (nx, ny) violates the collision radius of any OTHER robot.
#         Radius 1 means: Do not enter the 3x3 zone centered on another robot.
#         """
#         for r in all_robots:
#             if r.id == self.id:
#                 continue
#             # Chebyshev distance (square neighborhood) check
#             if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
#                 return False
#         return True

#     def choose_move_search(self, pher_rep: np.ndarray, all_robots: List['Robot'], 
#                           robot_radius: int, collision_radius: int = 1) -> Tuple[int, int]:
#         """
#         Choose next move based on utility function combining:
#         - Global novelty (cells never visited by anyone)
#         - Local novelty (cells not visited by this robot)
#         - Pheromone repulsion (avoid areas with high pheromone)
        
#         Args:
#             pher_rep: Repulsive pheromone field
#             all_robots: List of all robots for collision detection
#             robot_radius: Sensor radius R (used for KAPPA calculation)
#             collision_radius: Collision avoidance radius
            
#         Returns:
#             (nx, ny) next position
#         """
#         H, W = pher_rep.shape
        
#         # Calculate KAPPA based on robot radius: 2*R*(R+1) + 1
#         KAPPA = 20# * robot_radius * (robot_radius + 1) + 1
#         SIGMA = 20.0
        
#         # 1. Generate Valid Candidates (Collision & Bounds Checked)
#         candidates = []
#         # Check all 4 neighbors
#         for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
#             nx, ny = self.x + dx, self.y + dy
#             if 0 <= nx < W and 0 <= ny < H:
#                 if self.is_clear(nx, ny, all_robots, collision_radius):
#                     candidates.append((nx, ny))
        
#         # If completely blocked by robots/walls, stay put
#         if not candidates:
#             return (self.x, self.y)

#         # 2. Check Active Escape Mode (Priority Over Utility)
#         if self._escape_steps > 0:
#             self._escape_steps -= 1
#             # Random move to break corner traps
#             return random.choice(candidates)

#         # 3. Compute Utilities
#         scored_moves = []
        
#         for (nx, ny) in candidates:
#             # Lookahead: What will I see if I move to (nx, ny)?
#             u1_val = 0      # Global Novelty
#             u2_val = 0      # Local Novelty
#             pher_sum = 0.0  # Pheromone Sum
            
#             # Scan the sensor footprint of the CANDIDATE position
#             for dy in range(-robot_radius, robot_radius + 1):
#                 for dx in range(-robot_radius, robot_radius + 1):
#                     if abs(dx) + abs(dy) <= robot_radius:
#                         cx, cy = nx + dx, ny + dy
#                         if 0 <= cx < W and 0 <= cy < H:
#                             # Cache values to avoid repeated array access
#                             is_visited = self.local_covered[cy, cx]
#                             p_val = pher_rep[cy, cx]
                            
#                             pher_sum += p_val
                            
#                             if not is_visited:
#                                 u2_val += 1
#                                 if p_val == 0.0:  # Approx 0 check
#                                     u1_val += 1
                                
            
#             # Calculate Psi terms
#             term1 = float(u1_val)
#             term2 = float(u2_val) / KAPPA
            
            
#             # u3 is negative sum (Repulsion)
#             # Formula: (tanh(u3/sigma) + 1) / k^2
#             # Since u3 is negative, we pass -pher_sum
#             u3_input = -pher_sum
#             term3 = (np.tanh(u3_input / SIGMA) + 1.0) #/ (KAPPA ** 2)
#             print(f'term1:{term1}')
#             print(f'term2:{term2}')
#             print(f'pher_sum:{pher_sum}')
#             print(f'term3:{term3}')
#             print()
#             psi = term1 + term2 + term3
            
#             scored_moves.append({
#                 'move': (nx, ny),
#                 'score': psi,
#                 'u1_raw': u1_val,
#                 'u2_raw': u2_val
#             })

#         # 4. Selection
#         best = max(scored_moves, key=lambda x: x['score'])

#         # 5. Stagnation Check
#         if best['u1_raw'] == 0 and best['u2_raw'] == 0:
#             self._escape_steps = W//20  # Set countdown (k const)
            
#             # Trigger escape immediately by taking a random move NOW
#             # This prevents wasting a step staying in the stuck spot
#             return random.choice(candidates)
        
#         return best['move']

#     def step(self, pher_rep: np.ndarray, all_robots: List['Robot'], 
#              robot_radius: int, collision_radius: int = 1):
#         """Execute one step: choose move and update position."""
#         if self.failed:
#             return
        
#         nx, ny = self.choose_move_search(pher_rep, all_robots, robot_radius, collision_radius)
#         self.x, self.y = nx, ny

#     def deposit_pheromone(self, pher: np.ndarray, amount: float, r: int = 5):
#         """Deposit repulsive pheromone in neighborhood."""
#         deposit_uniform(pher, self.x, self.y, amount, r)


# This implementation has too many parameters to tune. DO NOT LIKE THIS.
# import numpy as np
# from dataclasses import dataclass, field
# from typing import List, Optional, Tuple
# from .pheromone import deposit_uniform


# def footprint_offsets(radius: int, shape: str = "manhattan"):
#     """Offsets for sensor footprint."""
#     offsets = []
#     for dy in range(-radius, radius + 1):
#         for dx in range(-radius, radius + 1):
#             if shape == "manhattan":
#                 ok = (abs(dx) + abs(dy) <= radius)
#             else:  # "euclidean"
#                 ok = (dx * dx + dy * dy <= radius * radius)
#             if ok:
#                 offsets.append((dx, dy))
#     return offsets


# @dataclass
# class Robot:
#     id: int
#     x: int
#     y: int
#     local_covered: np.ndarray
#     failed: bool = False

#     # --- tunable parameters ---
#     alpha: float = 1.0          # weight for global novelty proxy
#     beta: float = 0.3           # weight for local novelty
#     gamma: float = 1.0          # weight for pheromone repulsion
#     sigma: float = 15.0         # softness for tanh scaling
#     escape_len: int = 6         # how long to random-walk when stuck
#     collision_radius: int = 1   # Chebyshev radius to avoid other robots
#     footprint_shape: str = "manhattan"  # match your mark_visible/discovery
#     rng: Optional[np.random.Generator] = field(default=None, repr=False)

#     _escape_steps: int = 0
#     _stagnant_count: int = 0
#     stagnant_thresh: int = 5

#     def is_clear(self, nx: int, ny: int, all_robots: List['Robot']) -> bool:
#         for r in all_robots:
#             if r.id == self.id:
#                 continue
#             if max(abs(r.x - nx), abs(r.y - ny)) <= self.collision_radius:
#                 return False
#         return True

#     def choose_move_search(
#         self,
#         pher_rep: np.ndarray,
#         all_robots: List['Robot'],
#         robot_radius: int,
#     ) -> Tuple[int, int]:
#         H, W = pher_rep.shape

#         # Candidate moves: 4-neighbors only
#         candidates = []
#         for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
#             nx, ny = self.x + dx, self.y + dy
#             if 0 <= nx < W and 0 <= ny < H and self.is_clear(nx, ny, all_robots):
#                 candidates.append((nx, ny))
#         if not candidates:
#             return (self.x, self.y)

#         # Escape mode (short)
#         if self._escape_steps > 0:
#             self._escape_steps -= 1
#             if self.rng is not None:
#                 idx = int(self.rng.integers(0, len(candidates)))
#                 return candidates[idx]
#             return candidates[int(np.random.default_rng().integers(0, len(candidates)))]

#         offsets = footprint_offsets(robot_radius, self.footprint_shape)
#         K = len(offsets)  # correct footprint size

#         scored = []
#         for nx, ny in candidates:
#             u_global = 0
#             u_local = 0
#             pher_sum = 0.0

#             for dx, dy in offsets:
#                 cx, cy = nx + dx, ny + dy
#                 if 0 <= cx < W and 0 <= cy < H:
#                     pher_sum += float(pher_rep[cy, cx])
#                     if not self.local_covered[cy, cx]:
#                         u_local += 1
#                         # global novelty proxy: low pheromone implies globally less visited recently
#                         if pher_rep[cy, cx] < 1e-6:
#                             u_global += 1

#             # Normalize
#             term1 = self.alpha * float(u_global)
#             term2 = self.beta * (float(u_local) / max(1, K))

#             # Repulsion: higher pher_sum => lower score.
#             # Use tanh on pher_sum itself (not negative) so it decreases with pher_sum.
#             self.sigma = 0.2*W
#             repulsion = 1.0 - np.tanh(pher_sum / max(1e-6, self.sigma))
#             term3 = self.gamma * repulsion

#             score = term1 + term2 + term3
#             scored.append((score, nx, ny, u_global, u_local))

#         scored.sort(reverse=True, key=lambda t: t[0])
#         best_score, bx, by, u_g, u_l = scored[0]

#         # Stagnation logic: require multiple stagnant steps before escape
#         if u_g == 0 and u_l == 0:
#             self._stagnant_count += 1
#         else:
#             self._stagnant_count = 0

#         if self._stagnant_count >= self.stagnant_thresh:
#             self._stagnant_count = 0
#             self._escape_steps = self.escape_len  # uses grid-adaptive value set at construction
#             if self.rng is not None:
#                 idx = int(self.rng.integers(0, len(candidates)))
#                 return candidates[idx]
#             return candidates[int(np.random.default_rng().integers(0, len(candidates)))]

#         return (bx, by)

#     def step(self, pher_rep: np.ndarray, all_robots: List['Robot'], robot_radius: int):
#         if self.failed:
#             return
#         nx, ny = self.choose_move_search(pher_rep, all_robots, robot_radius)
#         self.x, self.y = nx, ny

#     def deposit_pheromone(self, pher: np.ndarray, amount: float, r: int = 5):
#         deposit_uniform(pher, self.x, self.y, amount, r)