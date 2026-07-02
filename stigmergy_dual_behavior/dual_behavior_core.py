from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
import argparse
import math
import os
import random
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.utilities import discover_targets_in_vnhood, mark_visible
import config_experiments
from stigmergy_common.pheromone import apply_decay, deposit_distance_signal, deposit_uniform


GRID_SIZES = (50, 100)
SEARCH_ROBOT_COUNTS = (1, 2, 3)
DISTANCE_SHELLS = (20, 30, 40, 50)
MIN_START_DISTANCE_TO_TARGET = 20
RUNS_PER_SHELL = config_experiments.RUNS_PER_SCENARIO
BASE_START_SEED = config_experiments.BASE_SEED * 500
BASE_ALGORITHM_SEED = config_experiments.BASE_SEED * 1000
ROBOT_RADIUS = config_experiments.ROBOT_RADIUS
COLLISION_RADIUS = 1
SPIRAL_LANE_SPACING = max(1, 2 * ROBOT_RADIUS)
PHER_DEPOSIT = 1.0
PHER_MIN = 1e-6


@dataclass(frozen=True)
class VariantSpec:
    name: str
    output_name: str
    rescue_mode: bool
    search_uses_repulsive: bool
    search_sign_flip: bool
    has_attractive_channel: bool
    merged_field: bool
    evaporate_search_field: bool


VARIANTS: Dict[str, VariantSpec] = {
    "repulsive_only": VariantSpec(
        name="repulsive_only",
        output_name="repulsive_only",
        rescue_mode=False,
        search_uses_repulsive=True,
        search_sign_flip=False,
        has_attractive_channel=False,
        merged_field=False,
        evaporate_search_field=True,
    ),
    "attractive_only": VariantSpec(
        name="attractive_only",
        output_name="attractive_only",
        rescue_mode=False,
        search_uses_repulsive=True,
        search_sign_flip=True,
        has_attractive_channel=False,
        merged_field=False,
        evaporate_search_field=False,
    ),
    "single_evaporative_merged": VariantSpec(
        name="single_evaporative_merged",
        output_name="single_evaporative_merged",
        rescue_mode=False,
        search_uses_repulsive=True,
        search_sign_flip=False,
        has_attractive_channel=False,
        merged_field=True,
        evaporate_search_field=True,
    ),
    "single_persistent_merged": VariantSpec(
        name="single_persistent_merged",
        output_name="single_persistent_merged",
        rescue_mode=False,
        search_uses_repulsive=True,
        search_sign_flip=False,
        has_attractive_channel=False,
        merged_field=True,
        evaporate_search_field=False,
    ),
    "sign_flip": VariantSpec(
        name="sign_flip",
        output_name="sign_flip",
        rescue_mode=False,
        search_uses_repulsive=True,
        search_sign_flip=True,
        has_attractive_channel=False,
        merged_field=False,
        evaporate_search_field=True,
    ),
    "stigmergy_search_and_rescue": VariantSpec(
        name="stigmergy_search_and_rescue",
        output_name="stigmergy_search_and_rescue",
        rescue_mode=True,
        search_uses_repulsive=True,
        search_sign_flip=False,
        has_attractive_channel=True,
        merged_field=False,
        evaporate_search_field=True,
    ),
}


def calculate_horizon(grid_size: int, n_search_robots: int) -> int:
    return config_experiments.calculate_horizon(grid_size, n_search_robots)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class SearchRescueRobot:
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
    _recent_search_positions: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=12), repr=False)
    _in_escape_mode: bool = False
    _escape_visited: Set[Tuple[int, int]] = field(default_factory=set, repr=False)

    def is_clear(self, nx: int, ny: int, all_robots: List["SearchRescueRobot"], radius: int = 1) -> bool:
        for r in all_robots:
            if r.id == self.id:
                continue
            if r.mode in {"FOUND", "ADVERTISE"}:
                continue
            if max(abs(r.x - nx), abs(r.y - ny)) <= radius:
                return False
        return True

    def choose_move_search(
        self,
        pher_search: np.ndarray,
        all_robots: List["SearchRescueRobot"],
        robot_radius: int,
        collision_radius: int = 1,
        *,
        use_pheromone: bool = True,
        sign_flip: bool = False,
    ) -> Tuple[int, int]:
        H, W = pher_search.shape
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
                footprint_pher = float(np.sum(pher_search[y_min:y_max, x_min:x_max])) if use_pheromone else 0.0
                visits = self._position_visits.get((nx, ny), 0)
                momentum_score = -1 if (self._last_move == (dx, dy)) else 0
                reversal_score = 1 if (self._last_move == (-dx, -dy)) else 0
                candidates.append(
                    {
                        "move": (nx, ny),
                        "action": (dx, dy),
                        "pher": footprint_pher,
                        "visits": visits,
                        "momentum": momentum_score,
                        "reversal": reversal_score,
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
            swarm_unvisited = (not use_pheromone) or pher_search[ny, nx] == 0.0
            self_unvisited = self._position_visits.get((nx, ny), 0) == 0
            if swarm_unvisited or self_unvisited:
                self._in_escape_mode = False
                self._escape_visited.clear()
            else:
                self._escape_visited.add(chosen["move"])
            return chosen["move"]

        if sign_flip and use_pheromone:
            non_recent = [c for c in candidates if c["move"] not in self._recent_search_positions]
            if non_recent:
                candidates = non_recent
            candidates.sort(key=lambda c: (c["reversal"], -round(c["pher"], 5), c["visits"], c["momentum"]))
        else:
            candidates.sort(key=lambda c: (round(c["pher"], 5), c["visits"], c["momentum"]))
        return candidates[0]["move"]

    def choose_move_follow(
        self,
        pher_attr: np.ndarray,
        all_robots: List["SearchRescueRobot"],
        collision_radius: int,
        sensing_radius: int = 5,
    ) -> Tuple[int, int]:
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
            length = float(np.sqrt(vec_x**2 + vec_y**2))
            if length > 0:
                vec_x /= length
                vec_y /= length

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

    def mark_target_found(
        self,
        target: Tuple[int, int],
        global_target_visits: Dict[Tuple[int, int], Set[int]],
        *,
        can_advertise: bool,
    ) -> None:
        self.x, self.y = target
        self.visited_targets.add(target)
        global_target_visits.setdefault(target, set()).add(self.id)
        if can_advertise:
            self.mode = "ADVERTISE"
            self.advertising_target = target
        else:
            self.mode = "FOUND"

    def step_search_or_follow(
        self,
        *,
        variant: VariantSpec,
        pher_search: np.ndarray,
        pher_follow: np.ndarray,
        targets: Set[Tuple[int, int]],
        all_robots: List["SearchRescueRobot"],
        global_target_visits: Dict[Tuple[int, int], Set[int]],
        robot_radius: int,
        collision_radius: int,
        pher_deposit: float,
        attractive_sensing_radius: int = 5,
        attraction_enabled: bool = True,
    ) -> None:
        if self.failed or self.mode in {"FOUND", "ADVERTISE"}:
            return

        target = next(iter(targets))
        if (self.x, self.y) in targets:
            can_advertise = variant.rescue_mode and not any(r.mode == "ADVERTISE" for r in all_robots)
            self.mark_target_found(target, global_target_visits, can_advertise=can_advertise)
            return

        old_x, old_y = self.x, self.y
        can_follow = variant.rescue_mode and variant.has_attractive_channel and attraction_enabled
        if can_follow and (self.mode == "FOLLOW" or self.senses_attractive(pher_follow, attractive_sensing_radius)):
            self.mode = "FOLLOW"
            nx, ny = self.choose_move_follow(pher_follow, all_robots, collision_radius, attractive_sensing_radius)
            if (nx, ny) == (self.x, self.y) and (nx, ny) not in targets:
                self.mode = "SEARCH"
            self.x, self.y = nx, ny
        else:
            nx, ny = self.choose_move_search(
                pher_search,
                all_robots,
                robot_radius,
                collision_radius,
                use_pheromone=variant.search_uses_repulsive,
                sign_flip=variant.search_sign_flip,
            )
            self.x, self.y = nx, ny
            self._last_move = (nx - old_x, ny - old_y)
            self._position_visits[(nx, ny)] = self._position_visits.get((nx, ny), 0) + 1
            self._recent_search_positions.append((nx, ny))
            if variant.search_uses_repulsive:
                deposit_uniform(pher_search, self.x, self.y, pher_deposit, robot_radius)

        if (self.x, self.y) in targets:
            can_advertise = variant.rescue_mode and not any(r.mode == "ADVERTISE" for r in all_robots)
            self.mark_target_found(target, global_target_visits, can_advertise=can_advertise)


def spiral_positions(
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    lane_spacing: int = 1,
) -> Iterable[Tuple[int, int]]:
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


def _evenly_spaced_cells(
    cells: List[Tuple[int, int]],
    target: Tuple[int, int],
    count: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    if len(cells) < count:
        raise ValueError(f"Need {count} cells but only found {len(cells)} candidates.")
    cx, cy = target
    ordered = sorted(cells, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    if count == 1:
        return [ordered[0]]

    def angle_distance(a: float, b: float) -> float:
        return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

    def is_spaced(pos: Tuple[int, int], selected: List[Tuple[int, int]]) -> bool:
        return all(max(abs(pos[0] - other[0]), abs(pos[1] - other[1])) > COLLISION_RADIUS for other in selected)

    starts: List[Tuple[int, int]] = []
    angle_offset = float(rng.uniform(0.0, 2.0 * math.pi / count))
    desired_angles = np.linspace(-math.pi, math.pi, count, endpoint=False) + angle_offset
    angle_by_cell = {pos: math.atan2(pos[1] - cy, pos[0] - cx) for pos in ordered}

    for desired_angle in desired_angles:
        candidates = sorted(ordered, key=lambda p: angle_distance(angle_by_cell[p], float(desired_angle)))
        for pos in candidates:
            if pos not in starts and is_spaced(pos, starts):
                starts.append(pos)
                break

    cursor = 0
    while len(starts) < count:
        remaining = [pos for pos in ordered if pos not in starts]
        if not remaining:
            break
        pos = max(
            remaining,
            key=lambda p: min(
                max(abs(p[0] - other[0]), abs(p[1] - other[1]))
                for other in starts
            )
            if starts
            else 0,
        )
        if is_spaced(pos, starts):
            starts.append(pos)
        else:
            pos = remaining[cursor % len(remaining)]
            if pos not in starts:
                starts.append(pos)
        cursor += 1
    return starts


def generate_spawn_position_sets(
    grid_size: int,
    distance_shells: Tuple[int, ...],
    n_search_robots: int,
    min_distance: int = MIN_START_DISTANCE_TO_TARGET,
) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    target = (grid_size // 2, grid_size // 2)
    all_cells = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) != target and manhattan((x, y), target) >= min_distance
    ]
    if len(all_cells) < n_search_robots:
        raise ValueError(f"Not enough cells at distance >= {min_distance} for grid_size={grid_size}")

    spawn_sets: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for requested_distance in distance_shells:
        start_seed = (
            BASE_START_SEED
            + (grid_size * 10000)
            + (n_search_robots * 1000)
            + (requested_distance * 10)
        )
        rng = np.random.default_rng(start_seed)
        exact_shell = [p for p in all_cells if manhattan(p, target) == requested_distance]
        if len(exact_shell) >= n_search_robots:
            starts = _evenly_spaced_cells(exact_shell, target, n_search_robots, rng)
        else:
            distance_candidates = sorted(
                {manhattan(p, target) for p in all_cells},
                key=lambda distance: (abs(distance - requested_distance), -distance),
            )
            starts = []
            for candidate_distance in distance_candidates:
                candidates = [p for p in all_cells if manhattan(p, target) == candidate_distance]
                if len(candidates) >= n_search_robots:
                    starts = _evenly_spaced_cells(candidates, target, n_search_robots, rng)
                    break
            if not starts:
                starts = _evenly_spaced_cells(all_cells, target, n_search_robots, rng)
        spawn_sets.append((requested_distance, start_seed, starts))
    return spawn_sets


def make_robots(grid_size: int, search_starts: List[Tuple[int, int]]) -> List[SearchRescueRobot]:
    return [
        SearchRescueRobot(
            id=i + 1,
            x=start[0],
            y=start[1],
            start_x=start[0],
            start_y=start[1],
            local_covered=np.zeros((grid_size, grid_size), dtype=bool),
            mode="SEARCH",
        )
        for i, start in enumerate(search_starts)
    ]


def _path_row(
    *,
    variant: VariantSpec,
    grid_size: int,
    n_search_robots: int,
    requested_distance_shell: int,
    spawn_position_id: int,
    start_seed: int,
    algorithm_seed: int,
    searcher: SearchRescueRobot,
    target: Tuple[int, int],
    step: int,
    attractive_detected: bool,
) -> Dict[str, object]:
    return {
        "variant": variant.name,
        "grid_size": grid_size,
        "n_search_robots": n_search_robots,
        "requested_distance_shell": requested_distance_shell,
        "spawn_position_id": spawn_position_id,
        "start_seed": start_seed,
        "algorithm_seed": algorithm_seed,
        "search_robot_id": searcher.id,
        "step": step,
        "search_x": searcher.x,
        "search_y": searcher.y,
        "mode": searcher.mode,
        "distance_to_target": manhattan((searcher.x, searcher.y), target),
        "attractive_detected": int(attractive_detected),
        "found_target": int(target in searcher.visited_targets),
    }


def _swarm_row(
    *,
    variant: VariantSpec,
    grid_size: int,
    n_search_robots: int,
    requested_distance_shell: int,
    spawn_position_id: int,
    start_seed: int,
    algorithm_seed: int,
    searchers: List[SearchRescueRobot],
    target: Tuple[int, int],
    step: int,
    first_found: bool,
    all_found: bool,
) -> Dict[str, object]:
    distances = [manhattan((searcher.x, searcher.y), target) for searcher in searchers]
    return {
        "variant": variant.name,
        "grid_size": grid_size,
        "n_search_robots": n_search_robots,
        "requested_distance_shell": requested_distance_shell,
        "spawn_position_id": spawn_position_id,
        "start_seed": start_seed,
        "algorithm_seed": algorithm_seed,
        "step": step,
        "min_distance_to_target_over_time": min(distances),
        "mean_distance_to_target_over_time": float(np.mean(distances)),
        "first_found": int(first_found),
        "all_found": int(all_found),
        "n_found": sum(1 for searcher in searchers if target in searcher.visited_targets),
    }


def _deposit_advertisers(
    *,
    variant: VariantSpec,
    robots: List[SearchRescueRobot],
    advertiser_spirals: Dict[int, Iterable[Tuple[int, int]]],
    pher_follow: np.ndarray,
    target: Tuple[int, int],
    grid_size: int,
) -> None:
    if not variant.rescue_mode or not variant.has_attractive_channel:
        return
    for robot in robots:
        if robot.mode != "ADVERTISE" or robot.advertising_target != target:
            continue
        deposit_distance_signal(pher_follow, robot.x, robot.y, target[0], target[1], ROBOT_RADIUS)
        if robot.id not in advertiser_spirals:
            spiral_iter = spiral_positions(target[0], target[1], grid_size, grid_size, lane_spacing=SPIRAL_LANE_SPACING)
            next(spiral_iter, None)
            advertiser_spirals[robot.id] = spiral_iter
        try:
            robot.x, robot.y = next(advertiser_spirals[robot.id])
        except StopIteration:
            pass


def run_simulation(
    variant: VariantSpec,
    grid_size: int,
    n_search_robots: int,
    requested_distance_shell: int,
    spawn_position_id: int,
    start_seed: int,
    search_starts: List[Tuple[int, int]],
    algorithm_seed: int,
) -> Dict[str, object]:
    random.seed(algorithm_seed)
    W = H = grid_size
    target = (W // 2, H // 2)
    targets = {target}
    max_horizon = calculate_horizon(grid_size, n_search_robots)
    tau_decay = (grid_size**2) / 2

    robots = make_robots(grid_size, search_starts)
    pher_search = np.zeros((H, W), dtype=float)
    pher_follow = pher_search if variant.merged_field else np.zeros((H, W), dtype=float)
    covered = np.zeros((H, W), dtype=bool)
    found_targets = set()
    global_target_visits: Dict[Tuple[int, int], Set[int]] = {}
    advertiser_spirals: Dict[int, Iterable[Tuple[int, int]]] = {}

    initial_distances = {r.id: manhattan((r.x, r.y), target) for r in robots}
    previous_distances = dict(initial_distances)
    detection_step_by_id = {r.id: None for r in robots}
    detection_distance_by_id = {r.id: np.nan for r in robots}
    mode_switch_step_by_id = {r.id: None for r in robots}
    found_step_by_id = {r.id: None for r in robots}
    final_distance_by_id = dict(initial_distances)
    stagnation_count_by_id = {r.id: 0 for r in robots}
    post_detection_stagnation_count_by_id = {r.id: 0 for r in robots}

    first_found = False
    all_found = False
    first_found_step = np.nan
    all_found_step = max_horizon
    first_found_robot_id: object = ""

    path_rows = [
        _path_row(
            variant=variant,
            grid_size=grid_size,
            n_search_robots=n_search_robots,
            requested_distance_shell=requested_distance_shell,
            spawn_position_id=spawn_position_id,
            start_seed=start_seed,
            algorithm_seed=algorithm_seed,
            searcher=searcher,
            target=target,
            step=0,
            attractive_detected=False,
        )
        for searcher in robots
    ]
    swarm_rows = [
        _swarm_row(
            variant=variant,
            grid_size=grid_size,
            n_search_robots=n_search_robots,
            requested_distance_shell=requested_distance_shell,
            spawn_position_id=spawn_position_id,
            start_seed=start_seed,
            algorithm_seed=algorithm_seed,
            searchers=robots,
            target=target,
            step=0,
            first_found=False,
            all_found=False,
        )
    ]

    for step in range(1, max_horizon + 1):
        if variant.evaporate_search_field:
            apply_decay(pher_search, tau_decay, PHER_MIN)

        _deposit_advertisers(
            variant=variant,
            robots=robots,
            advertiser_spirals=advertiser_spirals,
            pher_follow=pher_follow,
            target=target,
            grid_size=grid_size,
        )
        attraction_enabled = variant.rescue_mode and any(r.mode == "ADVERTISE" for r in robots)

        for searcher in robots:
            if searcher.mode in {"FOUND", "ADVERTISE"}:
                distance = manhattan((searcher.x, searcher.y), target)
                final_distance_by_id[searcher.id] = distance
                path_rows.append(
                    _path_row(
                        variant=variant,
                        grid_size=grid_size,
                        n_search_robots=n_search_robots,
                        requested_distance_shell=requested_distance_shell,
                        spawn_position_id=spawn_position_id,
                        start_seed=start_seed,
                        algorithm_seed=algorithm_seed,
                        searcher=searcher,
                        target=target,
                        step=step,
                        attractive_detected=detection_step_by_id[searcher.id] is not None,
                    )
                )
                continue

            previous_mode = searcher.mode
            current_distance = manhattan((searcher.x, searcher.y), target)
            if (
                variant.rescue_mode
                and variant.has_attractive_channel
                and attraction_enabled
                and searcher.senses_attractive(pher_follow, ROBOT_RADIUS)
                and detection_step_by_id[searcher.id] is None
            ):
                detection_step_by_id[searcher.id] = step
                detection_distance_by_id[searcher.id] = current_distance

            mark_visible(searcher.local_covered, searcher.x, searcher.y, ROBOT_RADIUS)
            mark_visible(covered, searcher.x, searcher.y, ROBOT_RADIUS)
            target_visible = manhattan((searcher.x, searcher.y), target) <= ROBOT_RADIUS
            discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, ROBOT_RADIUS)
            if target_visible:
                can_advertise = variant.rescue_mode and not any(r.mode == "ADVERTISE" for r in robots)
                searcher.mark_target_found(target, global_target_visits, can_advertise=can_advertise)
            else:
                searcher.step_search_or_follow(
                    variant=variant,
                    pher_search=pher_search,
                    pher_follow=pher_follow,
                    targets=targets,
                    all_robots=robots,
                    global_target_visits=global_target_visits,
                    robot_radius=ROBOT_RADIUS,
                    collision_radius=COLLISION_RADIUS,
                    pher_deposit=PHER_DEPOSIT,
                    attractive_sensing_radius=ROBOT_RADIUS,
                    attraction_enabled=attraction_enabled,
                )

            if previous_mode == "SEARCH" and searcher.mode == "FOLLOW" and mode_switch_step_by_id[searcher.id] is None:
                mode_switch_step_by_id[searcher.id] = step

            mark_visible(searcher.local_covered, searcher.x, searcher.y, ROBOT_RADIUS)
            mark_visible(covered, searcher.x, searcher.y, ROBOT_RADIUS)
            target_visible = manhattan((searcher.x, searcher.y), target) <= ROBOT_RADIUS
            discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, ROBOT_RADIUS)
            if target_visible and target not in searcher.visited_targets:
                can_advertise = variant.rescue_mode and not any(r.mode == "ADVERTISE" for r in robots)
                searcher.mark_target_found(target, global_target_visits, can_advertise=can_advertise)

            new_distance = manhattan((searcher.x, searcher.y), target)
            final_distance_by_id[searcher.id] = new_distance
            if new_distance >= previous_distances[searcher.id]:
                stagnation_count_by_id[searcher.id] += 1
                if detection_step_by_id[searcher.id] is not None:
                    post_detection_stagnation_count_by_id[searcher.id] += 1
            previous_distances[searcher.id] = new_distance

            if target in searcher.visited_targets and found_step_by_id[searcher.id] is None:
                found_step_by_id[searcher.id] = step
                if not first_found:
                    first_found = True
                    first_found_step = step
                    first_found_robot_id = searcher.id

            path_rows.append(
                _path_row(
                    variant=variant,
                    grid_size=grid_size,
                    n_search_robots=n_search_robots,
                    requested_distance_shell=requested_distance_shell,
                    spawn_position_id=spawn_position_id,
                    start_seed=start_seed,
                    algorithm_seed=algorithm_seed,
                    searcher=searcher,
                    target=target,
                    step=step,
                    attractive_detected=detection_step_by_id[searcher.id] is not None,
                )
            )

        all_found = all(target in searcher.visited_targets for searcher in robots)
        swarm_rows.append(
            _swarm_row(
                variant=variant,
                grid_size=grid_size,
                n_search_robots=n_search_robots,
                requested_distance_shell=requested_distance_shell,
                spawn_position_id=spawn_position_id,
                start_seed=start_seed,
                algorithm_seed=algorithm_seed,
                searchers=robots,
                target=target,
                step=step,
                first_found=first_found,
                all_found=all_found,
            )
        )
        if all_found:
            all_found_step = step
            break

    optimal_manhattan_steps = min(initial_distances.values())
    rescue_delay = all_found_step - first_found_step if all_found and np.isfinite(first_found_step) else np.nan
    path_efficiency = optimal_manhattan_steps / all_found_step if all_found and all_found_step > 0 else np.nan

    robot_rows = []
    for searcher in robots:
        robot_found_step = found_step_by_id[searcher.id]
        robot_detection_step = detection_step_by_id[searcher.id]
        robot_time_from_detection = (
            robot_found_step - robot_detection_step
            if robot_found_step is not None and robot_detection_step is not None
            else np.nan
        )
        robot_rows.append(
            {
                "variant": variant.name,
                "grid_size": grid_size,
                "n_search_robots": n_search_robots,
                "requested_distance_shell": requested_distance_shell,
                "spawn_position_id": spawn_position_id,
                "start_seed": start_seed,
                "algorithm_seed": algorithm_seed,
                "target_x": target[0],
                "target_y": target[1],
                "search_robot_id": searcher.id,
                "final_mode": searcher.mode,
                "search_start_x": searcher.start_x,
                "search_start_y": searcher.start_y,
                "initial_distance": initial_distances[searcher.id],
                "time_to_attractive_detection": robot_detection_step if robot_detection_step is not None else np.nan,
                "distance_at_detection": detection_distance_by_id[searcher.id],
                "mode_switch_step": mode_switch_step_by_id[searcher.id]
                if mode_switch_step_by_id[searcher.id] is not None
                else np.nan,
                "found": int(robot_found_step is not None),
                "steps_to_target": robot_found_step if robot_found_step is not None else np.nan,
                "time_from_detection_to_target": robot_time_from_detection,
                "stagnation_count": stagnation_count_by_id[searcher.id],
                "post_detection_stagnation_count": post_detection_stagnation_count_by_id[searcher.id],
                "final_distance": final_distance_by_id[searcher.id],
                "target_arrival_distance": 0 if robot_found_step is not None else final_distance_by_id[searcher.id],
            }
        )

    return {
        "variant": variant.name,
        "grid_size": grid_size,
        "n_search_robots": n_search_robots,
        "target_x": target[0],
        "target_y": target[1],
        "requested_distance_shell": requested_distance_shell,
        "spawn_position_id": spawn_position_id,
        "start_seed": start_seed,
        "search_starts": ";".join(f"{x},{y}" for x, y in search_starts),
        "search_start_distances": ";".join(str(initial_distances[searcher.id]) for searcher in robots),
        "actual_min_start_distance": min(initial_distances.values()),
        "actual_mean_start_distance": float(np.mean(list(initial_distances.values()))),
        "algorithm_seed": algorithm_seed,
        "max_horizon": max_horizon,
        "first_found": int(first_found),
        "first_found_robot_id": first_found_robot_id,
        "first_found_step": first_found_step,
        "all_found": int(all_found),
        "steps_to_all_found": all_found_step,
        "rescue_delay": rescue_delay,
        "optimal_manhattan_steps": optimal_manhattan_steps,
        "excess_steps_over_optimal": all_found_step - optimal_manhattan_steps if all_found else np.nan,
        "slowdown_ratio": all_found_step / optimal_manhattan_steps if all_found and optimal_manhattan_steps > 0 else np.nan,
        "path_efficiency": path_efficiency,
        "n_found": sum(1 for searcher in robots if target in searcher.visited_targets),
        "_path_rows": path_rows,
        "_swarm_rows": swarm_rows,
        "_robot_rows": robot_rows,
    }


Task = Tuple[int, int, int, int, int, int, List[Tuple[int, int]], int]


def build_tasks() -> List[Task]:
    tasks: List[Task] = []
    order = 0
    for grid_size in GRID_SIZES:
        for n_search_robots in SEARCH_ROBOT_COUNTS:
            spawn_sets = generate_spawn_position_sets(grid_size, DISTANCE_SHELLS, n_search_robots)
            for spawn_position_id, (requested_distance_shell, start_seed, search_starts) in enumerate(spawn_sets, 1):
                for run_idx in range(1, RUNS_PER_SHELL + 1):
                    algorithm_seed = (
                        BASE_ALGORITHM_SEED
                        + (grid_size * 10000)
                        + (n_search_robots * 1000)
                        + (requested_distance_shell * 10)
                        + run_idx
                    )
                    tasks.append(
                        (
                            order,
                            grid_size,
                            n_search_robots,
                            requested_distance_shell,
                            spawn_position_id,
                            start_seed,
                            search_starts,
                            algorithm_seed,
                        )
                    )
                    order += 1
    return tasks


def _run_task(variant_name: str, task: Task) -> Tuple[int, Dict[str, object]]:
    variant = VARIANTS[variant_name]
    order, grid_size, n_search_robots, requested_distance_shell, spawn_position_id, start_seed, search_starts, algorithm_seed = task
    print(
        f"Running {variant.name}: grid={grid_size}, searchers={n_search_robots}, "
        f"shell={requested_distance_shell}, starts={search_starts}, seed={algorithm_seed}",
        flush=True,
    )
    return order, run_simulation(
        variant,
        grid_size,
        n_search_robots,
        requested_distance_shell,
        spawn_position_id,
        start_seed,
        search_starts,
        algorithm_seed,
    )


def write_results(variant: VariantSpec, rows: List[Dict[str, object]]) -> Path:
    output_dir = CURRENT_DIR / "E5" / f"{variant.output_name}_parallel_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = output_dir / "results.xlsx"

    private_columns = {"_path_rows", "_swarm_rows", "_robot_rows"}
    scalar_rows = [{k: v for k, v in row.items() if k not in private_columns} for row in rows]
    path_rows = [path_row for row in rows for path_row in row.get("_path_rows", [])]
    swarm_rows = [swarm_row for row in rows for swarm_row in row.get("_swarm_rows", [])]
    robot_rows = [robot_row for row in rows for robot_row in row.get("_robot_rows", [])]

    df = pd.DataFrame(scalar_rows)
    path_df = pd.DataFrame(path_rows)
    swarm_df = pd.DataFrame(swarm_rows)
    robot_df = pd.DataFrame(robot_rows)

    path_df.to_csv(output_dir / "search_robot_paths.csv", index=False)
    swarm_df.to_csv(output_dir / "swarm_convergence.csv", index=False)
    robot_df.to_csv(output_dir / "search_robot_metrics.csv", index=False)

    summary = (
        df.groupby(["variant", "grid_size", "n_search_robots", "requested_distance_shell"], as_index=False)
        .agg(
            total_runs=("steps_to_all_found", "size"),
            first_found_rate=("first_found", "mean"),
            all_found_rate=("all_found", "mean"),
            avg_first_found_step=("first_found_step", "mean"),
            avg_steps_to_all_found=("steps_to_all_found", "mean"),
            avg_rescue_delay=("rescue_delay", "mean"),
            avg_n_found=("n_found", "mean"),
            avg_slowdown_ratio=("slowdown_ratio", "mean"),
            avg_path_efficiency=("path_efficiency", "mean"),
        )
    )

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")
        robot_df.to_excel(writer, index=False, sheet_name="robot_metrics")
        for sheet_name, frame in [("detailed", df), ("summary", summary), ("robot_metrics", robot_df)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx - 1, idx - 1, min(max_len + 2, 60))

    return xlsx_path


def run_experiments_for_variant(variant_name: str, workers: Optional[int] = None) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    variant = VARIANTS[variant_name]
    tasks = build_tasks()
    max_workers = workers or os.cpu_count() or 1
    completed: Dict[int, Dict[str, object]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, variant_name, task) for task in tasks]
        for future in as_completed(futures):
            order, result = future.result()
            completed[order] = result
            print(
                f"Completed {len(completed)}/{len(tasks)}: "
                f"all_found={result['all_found']}, n_found={result['n_found']}, "
                f"steps={result['steps_to_all_found']}, shell={result['requested_distance_shell']}, "
                f"searchers={result['n_search_robots']}",
                flush=True,
            )

    rows = [completed[i] for i in range(len(tasks))]
    xlsx_path = write_results(variant, rows)
    print(f"\nResults saved to: {xlsx_path}")
    print("Additional CSVs saved beside the workbook: search_robot_paths.csv, swarm_convergence.csv, search_robot_metrics.csv")


def main_experiment(variant_name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Defaults to CPU count.")
    args = parser.parse_args()
    run_experiments_for_variant(variant_name, workers=args.workers)
