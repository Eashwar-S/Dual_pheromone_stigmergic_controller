from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Robot:
    """Memoryless Robotarium random-walk policy."""
    id: int
    x: int
    y: int
    robot_radius: int
    collision_radius: int
    failed: bool = False

    def is_clear(self, nx: int, ny: int, all_robots: List["Robot"]) -> bool:
        """
        Checks if moving to (nx, ny) violates the collision radius of any other robot.
        """
        for r in all_robots:
            if r.id == self.id:
                continue
            if max(abs(r.x - nx), abs(r.y - ny)) <= self.collision_radius:
                return False
        return True

    def choose_move(self, rng: np.random.Generator, width: int, height: int,
                    all_robots: Optional[List["Robot"]] = None,
                    heading: Optional[float] = None) -> Tuple[int, int]:
        """Choose a random valid neighbor, biased by current unicycle heading."""
        candidates = []
        if self.y - 1 >= 0:
            candidates.append((self.x, self.y - 1))
        if self.y + 1 < height:
            candidates.append((self.x, self.y + 1))
        if self.x - 1 >= 0:
            candidates.append((self.x - 1, self.y))
        if self.x + 1 < width:
            candidates.append((self.x + 1, self.y))

        if all_robots is not None:
            candidates = [
                (nx, ny)
                for (nx, ny) in candidates
                if self.is_clear(nx, ny, all_robots)
            ]

        if not candidates:
            return self.x, self.y

        if heading is None:
            idx = rng.integers(0, len(candidates))
            return candidates[int(idx)]

        heading_vector = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        weights = []
        for nx, ny in candidates:
            move = np.array([nx - self.x, ny - self.y], dtype=float)
            norm = float(np.linalg.norm(move))
            if norm <= 0.0:
                weights.append(1.0)
                continue
            alignment = float(np.dot(heading_vector, move / norm))
            weights.append(max(0.0, 1.0 + alignment))

        w = np.array(weights, dtype=float)
        if np.all(w <= 0.0):
            w = np.ones_like(w)
        probs = w / w.sum()
        idx = rng.choice(len(candidates), p=probs)
        return candidates[int(idx)]

    def step(self, rng: np.random.Generator, width: int, height: int,
             all_robots: Optional[List["Robot"]] = None,
             heading: Optional[float] = None) -> None:
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        nx, ny = self.choose_move(rng, width, height, all_robots, heading)
        self.x, self.y = nx, ny
