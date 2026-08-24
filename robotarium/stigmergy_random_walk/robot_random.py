from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from robotarium_swarm_common import deposit_uniform


@dataclass
class Robot:
    """Repulsive-pheromone random walk with forward-motion preference."""
    id: int
    x: int
    y: int
    robot_radius: int
    collision_radius: int
    local_covered: np.ndarray
    failed: bool = False
    _last_move: Optional[Tuple[int, int]] = None

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
                    all_robots: Optional[List["Robot"]] = None,
                    heading: Optional[float] = None) -> Tuple[int, int]:
        """Choose a random forward-biased move away from repulsive pheromone."""
        height, width = pher.shape

        candidates = [
            (self.x + dx, self.y + dy)
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
            if 0 <= self.x + dx < width and 0 <= self.y + dy < height
        ]

        if all_robots is not None:
            candidates = [
                (nx, ny)
                for (nx, ny) in candidates
                if self.is_clear(nx, ny, all_robots)
            ]

        if not candidates:
            return self.x, self.y

        # Avoid an immediate 180-degree reversal when another collision-free
        # direction is available.
        if self._last_move is not None:
            reverse = (-self._last_move[0], -self._last_move[1])
            non_reversing = [
                (nx, ny)
                for nx, ny in candidates
                if (nx - self.x, ny - self.y) != reverse
            ]
            if non_reversing:
                candidates = non_reversing

        heading_vector = None
        if heading is not None:
            heading_vector = np.array([np.cos(heading), np.sin(heading)], dtype=float)
            forward_candidates = []
            for nx, ny in candidates:
                move = np.array([nx - self.x, ny - self.y], dtype=float)
                if float(np.dot(heading_vector, move)) >= -1e-9:
                    forward_candidates.append((nx, ny))
            if forward_candidates:
                candidates = forward_candidates

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
            if heading_vector is not None:
                move = np.array([nx - self.x, ny - self.y], dtype=float)
                alignment = float(np.dot(heading_vector, move))
                desirability *= max(0.05, 1.0 + alignment)
            weights.append(desirability)

        w = np.array(weights, dtype=float)
        if np.all(w <= 0):
            w = np.ones_like(w)

        probs = w / w.sum()
        idx = rng.choice(len(pool), p=probs)
        return pool[int(idx)]

    def step(self, pher: np.ndarray, rng: np.random.Generator,
             all_robots: Optional[List["Robot"]] = None,
             heading: Optional[float] = None) -> None:
        """Execute one step: choose move and update position."""
        if self.failed:
            return
        old_x, old_y = self.x, self.y
        nx, ny = self.choose_move(pher, rng, all_robots, heading)
        self.x, self.y = nx, ny
        self._last_move = (nx - old_x, ny - old_y)

    def deposit_pheromone(self, pher: np.ndarray, amount: float) -> None:
        """Deposit pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, robot_radius=self.robot_radius)
