from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random

import numpy as np

from robotarium_swarm_common import deposit_uniform


@dataclass
class Robot:
    """Robotarium-local search robot with unicycle-aware waypoint preference."""
    id: int
    x: int
    y: int
    local_covered: np.ndarray
    failed: bool = False

    _last_move: Optional[Tuple[int, int]] = None
    _position_visits: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    _in_escape_mode: bool = False
    _escape_visited: Set[Tuple[int, int]] = field(default_factory=set, repr=False)

    def is_clear(self, nx: int, ny: int, all_robots: List["Robot"], radius: int = 1) -> bool:
        """Checks if moving to (nx, ny) violates the collision radius of any other robot."""
        for r in all_robots:
            if r.id == self.id:
                continue
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    def turn_cost(self, action: Tuple[int, int]) -> int:
        if self._last_move is None:
            return 0
        if action == self._last_move:
            return 0
        if action == (-self._last_move[0], -self._last_move[1]):
            return 3
        return 1

    def is_dynamics_feasible(self, nx: int, ny: int, height: int, width: int,
                             robot_radius: int) -> bool:
        margin = robot_radius
        return margin <= nx < width - margin and margin <= ny < height - margin

    def choose_move_search(self, pher_rep: np.ndarray, all_robots: List["Robot"],
                           robot_radius: int, collision_radius: int = 1) -> Tuple[int, int]:
        height, width = pher_rep.shape

        if not self._position_visits:
            self._position_visits[(self.x, self.y)] = 1

        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(actions)
        candidates = []

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < width and 0 <= ny < height and self.is_clear(nx, ny, all_robots, collision_radius):
                y_min, y_max = max(0, ny - robot_radius), min(height, ny + robot_radius + 1)
                x_min, x_max = max(0, nx - robot_radius), min(width, nx + robot_radius + 1)
                footprint_pher = np.sum(pher_rep[y_min:y_max, x_min:x_max])

                candidates.append({
                    "move": (nx, ny),
                    "action": (dx, dy),
                    "pher": footprint_pher,
                    "visits": self._position_visits.get((nx, ny), 0),
                    "turn": self.turn_cost((dx, dy)),
                    "feasible": self.is_dynamics_feasible(nx, ny, height, width, robot_radius),
                })

        if not candidates:
            return self.x, self.y

        feasible_candidates = [c for c in candidates if c["feasible"]]
        if feasible_candidates:
            candidates = feasible_candidates

        all_visited = all(c["visits"] > 0 for c in candidates)
        if all_visited and not self._in_escape_mode:
            self._in_escape_mode = True
            self._escape_visited.clear()
            self._escape_visited.add((self.x, self.y))

        if self._in_escape_mode:
            escape_candidates = [c for c in candidates if c["move"] not in self._escape_visited]
            if not escape_candidates:
                self._escape_visited.clear()
                self._escape_visited.add((self.x, self.y))
                escape_candidates = candidates

            best_turn = min(c["turn"] for c in escape_candidates)
            turn_candidates = [c for c in escape_candidates if c["turn"] == best_turn]
            chosen = random.choice(turn_candidates)
            nx, ny = chosen["move"]

            swarm_unvisited = pher_rep[ny, nx] == 0.0
            self_unvisited = self._position_visits.get((nx, ny), 0) == 0

            if swarm_unvisited or self_unvisited:
                self._in_escape_mode = False
                self._escape_visited.clear()
            else:
                self._escape_visited.add(chosen["move"])

            return chosen["move"]

        candidates.sort(key=lambda c: (round(c["pher"], 5), c["visits"], c["turn"]))
        return candidates[0]["move"]

    def step(self, pher_rep: np.ndarray, all_robots: List["Robot"],
             robot_radius: int, collision_radius: int = 1) -> None:
        """Execute one step: choose move and update position."""
        if self.failed:
            return

        old_x, old_y = self.x, self.y
        nx, ny = self.choose_move_search(pher_rep, all_robots, robot_radius, collision_radius)

        self.x, self.y = nx, ny
        self._last_move = (nx - old_x, ny - old_y)
        self._position_visits[(nx, ny)] = self._position_visits.get((nx, ny), 0) + 1

    def deposit_pheromone(self, pher: np.ndarray, amount: float, r: int = 5) -> None:
        """Deposit repulsive pheromone in neighborhood."""
        deposit_uniform(pher, self.x, self.y, amount, r)
