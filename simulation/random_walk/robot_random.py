import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import sys
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent                        
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Robot:
    """Robot with private local map using memoryless random walk."""
    id: int
    x: int
    y: int
    robot_radius: int
    collision_radius: int
    local_covered: np.ndarray
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

    def choose_move(self, rng: np.random.Generator,
                    all_robots: List['Robot']) -> Tuple[int, int]:
        """Choose a uniformly random collision-free neighboring cell."""
        H, W = self.local_covered.shape

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

        idx = rng.integers(len(candidates))
        return candidates[idx]

    def step(self, rng: np.random.Generator, all_robots: List['Robot'] = None):
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        nx, ny = self.choose_move(rng, all_robots)
        self.x, self.y = nx, ny
