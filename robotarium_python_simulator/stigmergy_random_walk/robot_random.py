from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from robotarium_swarm_common import deposit_uniform


@dataclass
class Robot:
    """Robotarium-local copy of the biased random-walk robot policy."""
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

    def choose_move(self, pher: np.ndarray, rng: np.random.Generator,
                    all_robots: List["Robot"]) -> Tuple[int, int]:
        """Choose next move biased away from pheromone and toward uncovered cells."""
        height, width = pher.shape

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

        uncovered_candidates = [
            (nx, ny)
            for (nx, ny) in candidates
            if not self.local_covered[ny, nx]
        ]

        pool = uncovered_candidates if uncovered_candidates else candidates

        weights = []
        for nx, ny in pool:
            p = max(pher[ny, nx], 0.0)
            desirability = 1.0 / (1.0 + p)
            weights.append(desirability)

        w = np.array(weights, dtype=float)
        if np.all(w <= 0):
            w = np.ones_like(w)

        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[idx]

    def step(self, pher: np.ndarray, rng: np.random.Generator,
             all_robots: List["Robot"] = None) -> None:
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        nx, ny = self.choose_move(pher, rng, all_robots)
        self.x, self.y = nx, ny

    def deposit_pheromone(self, pher: np.ndarray, amount: float) -> None:
        """Deposit pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, robot_radius=self.robot_radius)
