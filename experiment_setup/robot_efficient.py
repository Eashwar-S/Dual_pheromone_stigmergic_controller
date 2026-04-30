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
        # print("len of best_u2 =============================", len(best_u2))
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
        # print("costs ========================", costs)
        # print()
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
