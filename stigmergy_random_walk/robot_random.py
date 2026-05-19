import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import sys
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent                        
sys.path.insert(0, str(PROJECT_ROOT))
from stigmergy_common.pheromone import deposit_uniform


@dataclass
class Robot:
    """Robot with private local map using biased random walk avoiding pheromone."""
    id: int
    x: int
    y: int
    robot_radius: int
    collision_radius: int
    local_covered: np.ndarray
    # last_move: Optional[Tuple[int, int]] = None # Memorly-less random walk
    failed: bool = False

    def is_clear(self, nx: int, ny: int, all_robots: List['Robot']) -> bool:
        """
        Checks if moving to (nx, ny) violates the collision radius of any OTHER robot.
        Radius 1 means: Do not enter the 3x3 zone centered on another robot.
        """
        for r in all_robots:
            if r.id == self.id:
                continue
            # Chebyshev distance (square neighborhood) check
            if max(abs(r.x - nx), abs(r.y - ny)) <= self.collision_radius:
                return False
        return True

    def choose_move(self, pher: np.ndarray, rng: np.random.Generator,
                           all_robots: List['Robot']) -> Tuple[int, int]:
        """Choose next move biased away from pheromone and toward uncovered cells."""
        H, W = pher.shape

        candidates = []
        if self.y - 1 >= 0:
            candidates.append((self.x, self.y - 1))
        if self.y + 1 < H:
            candidates.append((self.x, self.y + 1))
        if self.x - 1 >= 0:
            candidates.append((self.x - 1, self.y))
        if self.x + 1 < W:
            candidates.append((self.x + 1, self.y))

        # Filter out candidates that would collide with other robots
        if all_robots is not None:
            candidates = [
                (nx, ny)
                for (nx, ny) in candidates
                if self.is_clear(nx, ny, all_robots)
            ]

        if not candidates:
            return self.x, self.y

        # Prefer locally uncovered neighbors
        uncovered_candidates = [
            (nx, ny)
            for (nx, ny) in candidates
            if not self.local_covered[ny, nx]
        ]

        pool = uncovered_candidates if uncovered_candidates else candidates

        # Weight directions of exploration biasing towards lower pheromone direction
        weights = []
        for nx, ny in pool:
            p = max(pher[ny, nx], 0.0)
            desirability = 1.0 / (1.0 + p)  # main logic
            weights.append(desirability)

        w = np.array(weights, dtype=float)

        if np.all(w <= 0):
            w = np.ones_like(w)

        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)

        return pool[idx]

    def step(self, pher: np.ndarray, rng: np.random.Generator, all_robots: List['Robot'] = None):
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        nx, ny = self.choose_move(pher, rng, all_robots)
        # self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny

    def deposit_pheromone(self, pher: np.ndarray, amount: float):
        """Deposit pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, robot_radius=self.robot_radius)
