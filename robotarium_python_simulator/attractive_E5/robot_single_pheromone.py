from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random

import numpy as np

from robotarium_swarm_common import deposit_uniform


@dataclass
class Robot:
    """Search-only robot using a shared attractive pheromone field."""

    id: int
    x: int
    y: int
    local_covered: np.ndarray
    mode: str = "SEARCH"
    has_found_target: bool = False
    target_found_step: Optional[int] = None

    _last_move: Optional[Tuple[int, int]] = None
    _position_visits: Dict[Tuple[int, int], int] = field(
        default_factory=dict,
        repr=False,
    )
    _in_escape_mode: bool = False
    _escape_visited: Set[Tuple[int, int]] = field(
        default_factory=set,
        repr=False,
    )

    def is_clear(
        self,
        nx: int,
        ny: int,
        all_robots: List["Robot"],
        radius: int = 1,
    ) -> bool:
        """Return whether a grid cell respects the inter-robot clearance."""
        for robot in all_robots:
            if robot.id == self.id:
                continue
            if max(abs(robot.x - nx), abs(robot.y - ny)) <= radius:
                return False
        return True

    def turn_cost(self, action: Tuple[int, int]) -> int:
        if self._last_move is None or action == self._last_move:
            return 0
        if action == (-self._last_move[0], -self._last_move[1]):
            return 3
        return 1

    @staticmethod
    def is_dynamics_feasible(
        nx: int,
        ny: int,
        height: int,
        width: int,
        robot_radius: int,
    ) -> bool:
        return (
            robot_radius <= nx < width - robot_radius
            and robot_radius <= ny < height - robot_radius
        )

    def choose_move_search(
        self,
        pheromone: np.ndarray,
        all_robots: List["Robot"],
        robot_radius: int,
        collision_radius: int = 1,
        step_size: int = 1,
    ) -> Tuple[int, int]:
        """Prefer directions with stronger projected attractive pheromone."""
        height, width = pheromone.shape
        self.mode = "SEARCH"

        if not self._position_visits:
            self._position_visits[(self.x, self.y)] = 1

        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(actions)
        candidates = []

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not self.is_clear(nx, ny, all_robots, collision_radius):
                continue

            waypoint_x = int(np.clip(
                self.x + dx * max(1, step_size),
                robot_radius,
                width - 1 - robot_radius,
            ))
            waypoint_y = int(np.clip(
                self.y + dy * max(1, step_size),
                robot_radius,
                height - 1 - robot_radius,
            ))
            y_min = max(0, waypoint_y - robot_radius)
            y_max = min(height, waypoint_y + robot_radius + 1)
            x_min = max(0, waypoint_x - robot_radius)
            x_max = min(width, waypoint_x + robot_radius + 1)
            local_patch = self.local_covered[y_min:y_max, x_min:x_max]
            candidates.append({
                "move": (nx, ny),
                "waypoint": (waypoint_x, waypoint_y),
                "uncovered": int(np.count_nonzero(~local_patch)),
                "pheromone": float(
                    np.sum(pheromone[y_min:y_max, x_min:x_max])
                ),
                "visits": self._position_visits.get(
                    (waypoint_x, waypoint_y),
                    0,
                ),
                "reverse": (
                    self._last_move is not None
                    and (dx, dy) == (-self._last_move[0], -self._last_move[1])
                ),
                "turn": self.turn_cost((dx, dy)),
                "feasible": self.is_dynamics_feasible(
                    nx,
                    ny,
                    height,
                    width,
                    robot_radius,
                ),
            })

        if not candidates:
            return self.x, self.y

        feasible = [candidate for candidate in candidates if candidate["feasible"]]
        if feasible:
            candidates = feasible

        if (
            not self._in_escape_mode
            and all(candidate["visits"] > 0 for candidate in candidates)
        ):
            self._in_escape_mode = True
            self._escape_visited = {(self.x, self.y)}

        if self._in_escape_mode:
            escape_candidates = [
                candidate
                for candidate in candidates
                if candidate["move"] not in self._escape_visited
            ]
            if not escape_candidates:
                self._escape_visited = {(self.x, self.y)}
                escape_candidates = candidates

            best_turn = min(
                candidate["turn"] for candidate in escape_candidates
            )
            turn_candidates = [
                candidate
                for candidate in escape_candidates
                if candidate["turn"] == best_turn
            ]
            chosen = random.choice(turn_candidates)
            nx, ny = chosen["move"]
            waypoint_x, waypoint_y = chosen["waypoint"]

            if (
                pheromone[waypoint_y, waypoint_x] == 0.0
                or self._position_visits.get(
                    (waypoint_x, waypoint_y),
                    0,
                ) == 0
            ):
                self._in_escape_mode = False
                self._escape_visited.clear()
            else:
                self._escape_visited.add((nx, ny))
            return nx, ny

        # actions were shuffled first, so exact ties are broken reproducibly by
        # the seeded random order while the sort remains stable.
        candidates.sort(
            key=lambda candidate: (
                candidate["reverse"],
                -candidate["uncovered"],
                -round(candidate["pheromone"], 5),
                candidate["visits"],
                candidate["turn"],
            )
        )
        return candidates[0]["move"]

    def record_waypoint(
        self,
        start: Tuple[int, int],
        chosen: Tuple[int, int],
        waypoint: Tuple[int, int],
    ) -> None:
        """Update search memory after selecting a Robotarium waypoint."""
        self.mode = "SEARCH"
        self.x, self.y = waypoint
        self._last_move = (
            int(np.sign(chosen[0] - start[0])),
            int(np.sign(chosen[1] - start[1])),
        )
        self._position_visits[waypoint] = (
            self._position_visits.get(waypoint, 0) + 1
        )

    def deposit_pheromone(
        self,
        pheromone: np.ndarray,
        amount: float,
        robot_radius: int,
    ) -> None:
        """Deposit attractive evaporative pheromone at the robot footprint."""
        deposit_uniform(
            pheromone,
            self.x,
            self.y,
            amount,
            robot_radius,
        )
