from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
import random
import sys

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stigmergy_common.pheromone import deposit_distance_signal, deposit_uniform


@dataclass
class Robot:
    """Robot for rendezvous experiments with SEARCH/FOLLOW/ADVERTISE modes."""

    id: int
    x: int
    y: int
    local_covered: np.ndarray
    start_x: int
    start_y: int
    mode: str = "SEARCH"
    visited_targets: Set[Tuple[int, int]] = field(default_factory=set)
    advertising_target: Optional[Tuple[int, int]] = None
    follow_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=6))
    failed: bool = False

    _last_move: Optional[Tuple[int, int]] = None
    _position_visits: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    _in_escape_mode: bool = False
    _escape_visited: Set[Tuple[int, int]] = field(default_factory=set, repr=False)

    def is_clear(self, nx: int, ny: int, all_robots: List["Robot"], radius: int = 1) -> bool:
        """Checks if moving to (nx, ny) violates another robot's collision radius."""
        for r in all_robots:
            if r.id == self.id:
                continue
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    def choose_move_search(
        self,
        pher_rep: np.ndarray,
        all_robots: List["Robot"],
        robot_radius: int,
        collision_radius: int = 1,
    ) -> Tuple[int, int]:
        """Same repulsive pheromone search behavior as stigmergy_search/robot_efficient.py."""
        H, W = pher_rep.shape

        if not self._position_visits:
            self._position_visits[(self.x, self.y)] = 1

        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(actions)
        candidates = []

        for dx, dy in actions:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < W and 0 <= ny < H and self.is_clear(nx, ny, all_robots, collision_radius):
                y_min, y_max = max(0, ny - robot_radius), min(H, ny + robot_radius + 1)
                x_min, x_max = max(0, nx - robot_radius), min(W, nx + robot_radius + 1)
                footprint_pher = np.sum(pher_rep[y_min:y_max, x_min:x_max])

                visits = self._position_visits.get((nx, ny), 0)
                momentum_score = -1 if (self._last_move == (dx, dy)) else 0

                candidates.append(
                    {
                        "move": (nx, ny),
                        "action": (dx, dy),
                        "pher": footprint_pher,
                        "visits": visits,
                        "momentum": momentum_score,
                    }
                )

        if not candidates:
            return (self.x, self.y)

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

            chosen = random.choice(escape_candidates)
            nx, ny = chosen["move"]
            local_pher = pher_rep[ny, nx]
            swarm_unvisited = local_pher == 0.0
            self_unvisited = self._position_visits.get((nx, ny), 0) == 0

            if swarm_unvisited or self_unvisited:
                self._in_escape_mode = False
                self._escape_visited.clear()
            else:
                self._escape_visited.add(chosen["move"])

            return chosen["move"]

        candidates.sort(key=lambda c: (round(c["pher"], 5), c["visits"], c["momentum"]))
        best = candidates[0]
        return best["move"]

    def choose_move_follow(
        self,
        pher_attr: np.ndarray,
        all_robots: List["Robot"],
        collision_radius: int,
        sensing_radius: int = 5,
    ) -> Tuple[int, int]:
        """Choose a move by following the center of gravity of attractive pheromone."""
        H, W = pher_attr.shape

        vec_x, vec_y = 0.0, 0.0
        total_weight = 0.0
        found_signal = False

        for dy in range(-sensing_radius, sensing_radius + 1):
            for dx in range(-sensing_radius, sensing_radius + 1):
                if abs(dx) + abs(dy) <= sensing_radius:
                    cx, cy = self.x + dx, self.y + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        val = pher_attr[cy, cx]
                        if val > 1e-6:
                            found_signal = True
                            weight = val * val
                            vec_x += dx * weight
                            vec_y += dy * weight
                            total_weight += weight

        if not found_signal:
            self.mode = "SEARCH"
            return (self.x, self.y)

        if total_weight > 0:
            length = np.sqrt(vec_x**2 + vec_y**2)
            if length > 0:
                vec_x /= length
                vec_y /= length
        else:
            vec_x, vec_y = 0.0, 0.0

        candidates = []
        if self.y - 1 >= 0:
            candidates.append((0, -1, self.x, self.y - 1))
        if self.y + 1 < H:
            candidates.append((0, 1, self.x, self.y + 1))
        if self.x - 1 >= 0:
            candidates.append((-1, 0, self.x - 1, self.y))
        if self.x + 1 < W:
            candidates.append((1, 0, self.x + 1, self.y))

        random.shuffle(candidates)

        best_move = None
        best_score = -999.0

        for dx, dy, nx, ny in candidates:
            if (nx, ny) in self.follow_history:
                continue
            if not self.is_clear(nx, ny, all_robots, collision_radius):
                continue

            alignment = (dx * vec_x) + (dy * vec_y)
            intensity = pher_attr[ny, nx]
            score = alignment + (intensity * 0.1)

            if score > best_score:
                best_score = score
                best_move = (nx, ny)

        if best_move is None:
            self.follow_history.clear()
            max_p = -1.0
            fallback = (self.x, self.y)
            for _, _, nx, ny in candidates:
                if self.is_clear(nx, ny, all_robots, collision_radius) and pher_attr[ny, nx] > max_p:
                    max_p = pher_attr[ny, nx]
                    fallback = (nx, ny)
            return fallback

        self.follow_history.append(best_move)
        return best_move

    def senses_attractive(self, pher_attr: np.ndarray, sensing_radius: int = 5, threshold: float = 1e-5) -> bool:
        H, W = pher_attr.shape
        for dy in range(-sensing_radius, sensing_radius + 1):
            for dx in range(-sensing_radius, sensing_radius + 1):
                if abs(dx) + abs(dy) <= sensing_radius:
                    cx, cy = self.x + dx, self.y + dy
                    if 0 <= cx < W and 0 <= cy < H and pher_attr[cy, cx] > threshold:
                        return True
        return False

    def step_search_or_follow(
        self,
        pher_rep: np.ndarray,
        pher_attr: np.ndarray,
        targets: Set[Tuple[int, int]],
        all_robots: List["Robot"],
        global_target_visits: Dict[Tuple[int, int], Set[int]],
        robot_radius: int,
        collision_radius: int,
        pher_deposit: float,
        attractive_sensing_radius: int = 5,
    ) -> None:
        """Move in SEARCH until attractive pheromone is sensed, then FOLLOW it."""
        if self.failed or self.mode == "FOUND":
            return

        if (self.x, self.y) in targets:
            self.visited_targets.add((self.x, self.y))
            global_target_visits.setdefault((self.x, self.y), set()).add(self.id)
            self.mode = "FOUND"
            return

        old_x, old_y = self.x, self.y

        if self.mode == "FOLLOW" or self.senses_attractive(pher_attr, attractive_sensing_radius):
            self.mode = "FOLLOW"
            nx, ny = self.choose_move_follow(pher_attr, all_robots, collision_radius, attractive_sensing_radius)
            if (nx, ny) == (self.x, self.y) and (nx, ny) not in targets:
                self.mode = "SEARCH"
            self.x, self.y = nx, ny
        else:
            nx, ny = self.choose_move_search(pher_rep, all_robots, robot_radius, collision_radius)
            self.x, self.y = nx, ny
            self._last_move = (nx - old_x, ny - old_y)
            self._position_visits[(nx, ny)] = self._position_visits.get((nx, ny), 0) + 1
            self.deposit_repulsive(pher_rep, pher_deposit, robot_radius)

        if (self.x, self.y) in targets:
            self.visited_targets.add((self.x, self.y))
            global_target_visits.setdefault((self.x, self.y), set()).add(self.id)
            self.mode = "FOUND"

    def deposit_repulsive(self, pher_rep: np.ndarray, amount: float, robot_radius: int) -> None:
        deposit_uniform(pher_rep, self.x, self.y, amount, robot_radius)

    def deposit_attractive(
        self,
        pher_attr: np.ndarray,
        target_x: int,
        target_y: int,
        attractive_radius: int,
    ) -> None:
        deposit_distance_signal(pher_attr, self.x, self.y, target_x, target_y, attractive_radius)


def spiral_positions(
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    lane_spacing: int = 1,
) -> Iterable[Tuple[int, int]]:
    """Yield an outward square spiral with separated lanes.

    ``lane_spacing`` controls the distance between adjacent spiral arms. Use
    ``2 * robot_radius`` for minimal overlap between neighboring robot
    footprints, which avoids diagonal gaps in Manhattan-radius deposition.
    """
    lane_spacing = max(1, int(lane_spacing))
    x, y = center_x, center_y
    if 0 <= x < width and 0 <= y < height:
        yield x, y

    step_len = lane_spacing
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    while step_len <= max(width, height) * 2:
        for dir_idx, (dx, dy) in enumerate(directions):
            for _ in range(step_len):
                x += dx
                y += dy
                if 0 <= x < width and 0 <= y < height:
                    yield x, y
            if dir_idx % 2 == 1:
                step_len += lane_spacing
