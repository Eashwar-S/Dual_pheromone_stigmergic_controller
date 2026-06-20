from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
import random

import numpy as np

from robotarium_swarm_common import deposit_distance_signal, deposit_uniform


@dataclass
class Robot:
    """Robot using repulsive search and attractive target pheromones."""

    id: int
    x: int
    y: int
    local_covered: np.ndarray
    mode: str = "SEARCH"
    has_found_target: bool = False
    target_found_step: Optional[int] = None
    advertising_target: Optional[Tuple[int, int]] = None
    follow_history: Deque[Tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=6)
    )

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
        repulsive_pheromone: np.ndarray,
        all_robots: List["Robot"],
        robot_radius: int,
        collision_radius: int = 1,
    ) -> Tuple[int, int]:
        """Choose a low-pheromone search direction."""
        height, width = repulsive_pheromone.shape
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

            y_min = max(0, ny - robot_radius)
            y_max = min(height, ny + robot_radius + 1)
            x_min = max(0, nx - robot_radius)
            x_max = min(width, nx + robot_radius + 1)
            candidates.append({
                "move": (nx, ny),
                "pheromone": float(
                    np.sum(
                        repulsive_pheromone[
                            y_min:y_max,
                            x_min:x_max,
                        ]
                    )
                ),
                "visits": self._position_visits.get((nx, ny), 0),
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
            chosen = random.choice([
                candidate
                for candidate in escape_candidates
                if candidate["turn"] == best_turn
            ])
            nx, ny = chosen["move"]
            if (
                repulsive_pheromone[ny, nx] == 0.0
                or self._position_visits.get((nx, ny), 0) == 0
            ):
                self._in_escape_mode = False
                self._escape_visited.clear()
            else:
                self._escape_visited.add((nx, ny))
            return nx, ny

        candidates.sort(
            key=lambda candidate: (
                round(candidate["pheromone"], 5),
                candidate["visits"],
                candidate["turn"],
            )
        )
        return candidates[0]["move"]

    def senses_attractive(
        self,
        attractive_pheromone: np.ndarray,
        sensing_radius: int = 5,
        threshold: float = 1e-5,
    ) -> bool:
        height, width = attractive_pheromone.shape
        for dy in range(-sensing_radius, sensing_radius + 1):
            for dx in range(-sensing_radius, sensing_radius + 1):
                if abs(dx) + abs(dy) > sensing_radius:
                    continue
                cx, cy = self.x + dx, self.y + dy
                if (
                    0 <= cx < width
                    and 0 <= cy < height
                    and attractive_pheromone[cy, cx] > threshold
                ):
                    return True
        return False

    def choose_move_follow(
        self,
        attractive_pheromone: np.ndarray,
        all_robots: List["Robot"],
        collision_radius: int,
        sensing_radius: int = 5,
    ) -> Tuple[int, int]:
        """Follow the local center of mass of attractive pheromone."""
        height, width = attractive_pheromone.shape
        vec_x = 0.0
        vec_y = 0.0
        total_weight = 0.0

        for dy in range(-sensing_radius, sensing_radius + 1):
            for dx in range(-sensing_radius, sensing_radius + 1):
                if abs(dx) + abs(dy) > sensing_radius:
                    continue
                cx, cy = self.x + dx, self.y + dy
                if not (0 <= cx < width and 0 <= cy < height):
                    continue
                value = attractive_pheromone[cy, cx]
                if value <= 1e-6:
                    continue
                weight = value * value
                vec_x += dx * weight
                vec_y += dy * weight
                total_weight += weight

        if total_weight <= 0:
            self.mode = "SEARCH"
            return self.x, self.y

        length = float(np.hypot(vec_x, vec_y))
        if length > 0:
            vec_x /= length
            vec_y /= length

        candidates = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = self.x + dx, self.y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if not self.is_clear(nx, ny, all_robots, collision_radius):
                continue
            candidates.append({
                "move": (nx, ny),
                "alignment": dx * vec_x + dy * vec_y,
                "intensity": attractive_pheromone[ny, nx],
            })

        if not candidates:
            return self.x, self.y

        fresh = [
            candidate
            for candidate in candidates
            if candidate["move"] not in self.follow_history
        ]
        if fresh:
            candidates = fresh
        else:
            self.follow_history.clear()

        random.shuffle(candidates)
        chosen = max(
            candidates,
            key=lambda candidate: (
                candidate["alignment"],
                candidate["intensity"],
            ),
        )
        self.follow_history.append(chosen["move"])
        return chosen["move"]

    def choose_search_or_follow(
        self,
        repulsive_pheromone: np.ndarray,
        attractive_pheromone: np.ndarray,
        all_robots: List["Robot"],
        robot_radius: int,
        collision_radius: int,
        attractive_sensing_radius: int,
    ) -> Tuple[int, int]:
        if (
            self.mode == "FOLLOW"
            or self.senses_attractive(
                attractive_pheromone,
                attractive_sensing_radius,
            )
        ):
            self.mode = "FOLLOW"
            goal = self.choose_move_follow(
                attractive_pheromone,
                all_robots,
                collision_radius,
                attractive_sensing_radius,
            )
            if goal != (self.x, self.y):
                return goal

        self.mode = "SEARCH"
        return self.choose_move_search(
            repulsive_pheromone,
            all_robots,
            robot_radius,
            collision_radius,
        )

    def record_waypoint(
        self,
        start: Tuple[int, int],
        chosen: Tuple[int, int],
        waypoint: Tuple[int, int],
    ) -> None:
        self.x, self.y = waypoint
        self._last_move = (
            int(np.sign(chosen[0] - start[0])),
            int(np.sign(chosen[1] - start[1])),
        )
        self._position_visits[waypoint] = (
            self._position_visits.get(waypoint, 0) + 1
        )

    def deposit_repulsive(
        self,
        repulsive_pheromone: np.ndarray,
        amount: float,
        robot_radius: int,
    ) -> None:
        deposit_uniform(
            repulsive_pheromone,
            self.x,
            self.y,
            amount,
            robot_radius,
        )

    def deposit_attractive(
        self,
        attractive_pheromone: np.ndarray,
        target_x: int,
        target_y: int,
        attractive_radius: int,
    ) -> None:
        deposit_distance_signal(
            attractive_pheromone,
            self.x,
            self.y,
            target_x,
            target_y,
            attractive_radius,
        )


def spiral_positions(
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    lane_spacing: int = 1,
) -> Iterable[Tuple[int, int]]:
    """Yield cells on an outward square spiral around the target."""
    lane_spacing = max(1, int(lane_spacing))
    x, y = center_x, center_y
    if 0 <= x < width and 0 <= y < height:
        yield x, y

    step_len = lane_spacing
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while step_len <= max(width, height) * 2:
        for direction_index, (dx, dy) in enumerate(directions):
            for _ in range(step_len):
                x += dx
                y += dy
                if 0 <= x < width and 0 <= y < height:
                    yield x, y
            if direction_index % 2 == 1:
                step_len += lane_spacing
