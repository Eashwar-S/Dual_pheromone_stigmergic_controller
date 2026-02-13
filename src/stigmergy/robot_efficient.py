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
        """
        Choose next move based on utility function combining:
        - Global novelty (cells never visited by anyone)
        - Local novelty (cells not visited by this robot)
        - Pheromone repulsion (avoid areas with high pheromone)
        
        Args:
            pher_rep: Repulsive pheromone field
            all_robots: List of all robots for collision detection
            robot_radius: Sensor radius R (used for KAPPA calculation)
            collision_radius: Collision avoidance radius
            
        Returns:
            (nx, ny) next position
        """
        H, W = pher_rep.shape
        
        # Calculate KAPPA based on robot radius: 2*R*(R+1) + 1
        KAPPA = 2 * robot_radius * (robot_radius + 1) + 1
        SIGMA = 10.0
        
        # 1. Generate Valid Candidates (Collision & Bounds Checked)
        candidates = []
        # Check all 4 neighbors
        for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < W and 0 <= ny < H:
                if self.is_clear(nx, ny, all_robots, collision_radius):
                    candidates.append((nx, ny))
        
        # If completely blocked by robots/walls, stay put
        if not candidates:
            return (self.x, self.y)

        # 2. Check Active Escape Mode (Priority Over Utility)
        if self._escape_steps > 0:
            self._escape_steps -= 1
            # Random move to break corner traps
            return random.choice(candidates)

        # 3. Compute Utilities
        scored_moves = []
        
        for (nx, ny) in candidates:
            # Lookahead: What will I see if I move to (nx, ny)?
            u1_val = 0      # Global Novelty
            u2_val = 0      # Local Novelty
            pher_sum = 0.0  # Pheromone Sum
            
            # Scan the sensor footprint of the CANDIDATE position
            for dy in range(-robot_radius, robot_radius + 1):
                for dx in range(-robot_radius, robot_radius + 1):
                    if abs(dx) + abs(dy) <= robot_radius:
                        cx, cy = nx + dx, ny + dy
                        if 0 <= cx < W and 0 <= cy < H:
                            # Cache values to avoid repeated array access
                            is_visited = self.local_covered[cy, cx]
                            p_val = pher_rep[cy, cx]
                            
                            pher_sum += p_val
                            
                            if not is_visited:
                                u2_val += 1
                                if p_val < 1e-3:  # Approx 0 check
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
                'u1_raw': u1_val,
                'u2_raw': u2_val
            })

        # 4. Selection
        best = max(scored_moves, key=lambda x: x['score'])

        # 5. Stagnation Check
        if best['u1_raw'] == 0 and best['u2_raw'] == 0:
            self._escape_steps = 5  # Set countdown (k const)
            
            # Trigger escape immediately by taking a random move NOW
            # This prevents wasting a step staying in the stuck spot
            return random.choice(candidates)
        
        return best['move']

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
