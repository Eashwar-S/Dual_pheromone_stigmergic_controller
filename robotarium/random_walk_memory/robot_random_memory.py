from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Robot:
    """Random-walk policy with private memory of locally visited cells."""
    id: int
    x: int
    y: int
    robot_radius: int
    collision_radius: int
    local_covered: np.ndarray
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
        """Choose a valid neighbor, preferring cells this robot has not visited."""
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

        unvisited_candidates = [
            (nx, ny)
            for nx, ny in candidates
            if not self.local_covered[ny, nx]
        ]
        pool = unvisited_candidates if unvisited_candidates else candidates

        heading_vector = None
        if heading is not None:
            heading_vector = np.array(
                [np.cos(heading), np.sin(heading)],
                dtype=float,
            )

        weights = []
        for nx, ny in pool:
            weight = 1.0
            if heading_vector is not None:
                move = np.array([nx - self.x, ny - self.y], dtype=float)
                alignment = float(np.dot(heading_vector, move))
                weight *= max(0.05, 1.0 + alignment)
            weights.append(weight)

        w = np.array(weights, dtype=float)
        if np.all(w <= 0.0):
            w = np.ones_like(w)
        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[int(idx)]

    def step(self, rng: np.random.Generator, width: int, height: int,
             all_robots: Optional[List["Robot"]] = None,
             heading: Optional[float] = None) -> None:
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        nx, ny = self.choose_move(rng, width, height, all_robots, heading)
        self.x, self.y = nx, ny
