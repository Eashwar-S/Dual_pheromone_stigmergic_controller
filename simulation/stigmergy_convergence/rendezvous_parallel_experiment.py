from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
sys.path.insert(0, str(CURRENT_DIR))

from common.utilities import discover_targets_in_vnhood, mark_visible
import config_experiments
from robot_rendezvous import Robot, spiral_positions
from stigmergy_common.pheromone import apply_decay


GRID_SIZES = (50, 100)
SEARCH_ROBOT_COUNTS = (1, 2, 3)
DISTANCE_SHELLS = (20, 30, 40, 50)
MIN_START_DISTANCE_TO_TARGET = 20
RUNS_PER_SHELL = config_experiments.RUNS_PER_SCENARIO
BASE_ALGORITHM_SEED = config_experiments.BASE_SEED * 1000
ROBOT_RADIUS = config_experiments.ROBOT_RADIUS
COLLISION_RADIUS = 1
SPIRAL_LANE_SPACING = max(1, 2 * ROBOT_RADIUS)
PHER_DEPOSIT = 1.0
PHER_MIN = 1e-6


def calculate_horizon(grid_size: int) -> int:
    return config_experiments.calculate_horizon(grid_size, 2)


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _evenly_spaced_cells(cells: List[Tuple[int, int]], target: Tuple[int, int], count: int) -> List[Tuple[int, int]]:
    if len(cells) < count:
        raise ValueError(f"Need {count} cells but only found {len(cells)} candidates.")

    cx, cy = target
    ordered = sorted(cells, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    if count == 1:
        return [ordered[0]]

    indices = np.linspace(0, len(ordered) - 1, count, dtype=int)
    starts: List[Tuple[int, int]] = []
    for idx in indices:
        pos = ordered[int(idx)]
        if pos not in starts:
            starts.append(pos)

    cursor = 0
    while len(starts) < count:
        pos = ordered[cursor % len(ordered)]
        if pos not in starts:
            starts.append(pos)
        cursor += 1
    return starts


def generate_spawn_position_sets(
    grid_size: int,
    distance_shells: Tuple[int, ...],
    n_search_robots: int,
    min_distance: int = MIN_START_DISTANCE_TO_TARGET,
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Generate start sets by requested Manhattan distance shells."""
    target = (grid_size // 2, grid_size // 2)
    all_cells = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) != target and manhattan((x, y), target) >= min_distance
    ]
    if len(all_cells) < n_search_robots:
        raise ValueError(f"Not enough cells at distance >= {min_distance} for grid_size={grid_size}")

    spawn_sets: List[Tuple[int, List[Tuple[int, int]]]] = []
    for requested_distance in distance_shells:
        exact_shell = [p for p in all_cells if manhattan(p, target) == requested_distance]
        if len(exact_shell) >= n_search_robots:
            starts = _evenly_spaced_cells(exact_shell, target, n_search_robots)
        else:
            distance_candidates = sorted(
                {manhattan(p, target) for p in all_cells},
                key=lambda distance: (abs(distance - requested_distance), -distance),
            )
            starts = []
            for candidate_distance in distance_candidates:
                candidates = [p for p in all_cells if manhattan(p, target) == candidate_distance]
                if len(candidates) >= n_search_robots:
                    starts = _evenly_spaced_cells(candidates, target, n_search_robots)
                    break
            if not starts:
                starts = _evenly_spaced_cells(all_cells, target, n_search_robots)
        spawn_sets.append((requested_distance, starts))
    return spawn_sets


def make_robots(grid_size: int, search_starts: List[Tuple[int, int]], target: Tuple[int, int]) -> List[Robot]:
    advertiser = Robot(
        id=0,
        x=target[0],
        y=target[1],
        start_x=target[0],
        start_y=target[1],
        local_covered=np.zeros((grid_size, grid_size), dtype=bool),
        mode="ADVERTISE",
        advertising_target=target,
    )
    searchers = [
        Robot(
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
    return [advertiser, *searchers]


def _path_row(
    *,
    grid_size: int,
    n_search_robots: int,
    requested_distance_shell: int,
    spawn_position_id: int,
    algorithm_seed: int,
    searcher: Robot,
    target: Tuple[int, int],
    step: int,
    attractive_detected: bool,
) -> Dict[str, object]:
    return {
        "grid_size": grid_size,
        "n_search_robots": n_search_robots,
        "requested_distance_shell": requested_distance_shell,
        "spawn_position_id": spawn_position_id,
        "algorithm_seed": algorithm_seed,
        "search_robot_id": searcher.id,
        "step": step,
        "search_x": searcher.x,
        "search_y": searcher.y,
        "mode": searcher.mode,
        "distance_to_target": manhattan((searcher.x, searcher.y), target),
        "attractive_detected": int(attractive_detected),
    }


def _swarm_row(
    *,
    grid_size: int,
    n_search_robots: int,
    requested_distance_shell: int,
    spawn_position_id: int,
    algorithm_seed: int,
    searchers: List[Robot],
    target: Tuple[int, int],
    step: int,
    found: bool,
) -> Dict[str, object]:
    distances = [manhattan((searcher.x, searcher.y), target) for searcher in searchers]
    return {
        "grid_size": grid_size,
        "n_search_robots": n_search_robots,
        "requested_distance_shell": requested_distance_shell,
        "spawn_position_id": spawn_position_id,
        "algorithm_seed": algorithm_seed,
        "step": step,
        "min_distance_to_target_over_time": min(distances),
        "mean_distance_to_target_over_time": float(np.mean(distances)),
        "found": int(found),
    }


def run_simulation(
    grid_size: int,
    n_search_robots: int,
    requested_distance_shell: int,
    spawn_position_id: int,
    search_starts: List[Tuple[int, int]],
    algorithm_seed: int,
) -> Dict[str, object]:
    random.seed(algorithm_seed)

    W = H = grid_size
    target = (W // 2, H // 2)
    targets = {target}
    max_horizon = calculate_horizon(grid_size)
    tau_decay = (grid_size**2) / 2

    robots = make_robots(grid_size, search_starts, target)
    advertiser = robots[0]
    searchers = robots[1:]
    spiral_iter = spiral_positions(target[0], target[1], W, H, lane_spacing=SPIRAL_LANE_SPACING)
    next(spiral_iter)

    pher_rep = np.zeros((H, W), dtype=float)
    pher_attr = np.zeros((H, W), dtype=float)
    covered = np.zeros((H, W), dtype=bool)
    found_targets = set()
    global_target_visits = {}

    initial_distances = {searcher.id: manhattan((searcher.x, searcher.y), target) for searcher in searchers}
    previous_distances = dict(initial_distances)
    detection_step_by_id = {searcher.id: None for searcher in searchers}
    detection_distance_by_id = {searcher.id: np.nan for searcher in searchers}
    mode_switch_step_by_id = {searcher.id: None for searcher in searchers}
    found_step_by_id = {searcher.id: None for searcher in searchers}
    final_distance_by_id = dict(initial_distances)
    stagnation_count_by_id = {searcher.id: 0 for searcher in searchers}
    post_detection_stagnation_count_by_id = {searcher.id: 0 for searcher in searchers}

    found = False
    steps_to_target = max_horizon
    found_robot_id: object = ""
    path_rows = [
        _path_row(
            grid_size=grid_size,
            n_search_robots=n_search_robots,
            requested_distance_shell=requested_distance_shell,
            spawn_position_id=spawn_position_id,
            algorithm_seed=algorithm_seed,
            searcher=searcher,
            target=target,
            step=0,
            attractive_detected=False,
        )
        for searcher in searchers
    ]
    swarm_rows = [
        _swarm_row(
            grid_size=grid_size,
            n_search_robots=n_search_robots,
            requested_distance_shell=requested_distance_shell,
            spawn_position_id=spawn_position_id,
            algorithm_seed=algorithm_seed,
            searchers=searchers,
            target=target,
            step=0,
            found=False,
        )
    ]

    for step in range(1, max_horizon + 1):
        apply_decay(pher_rep, tau_decay, PHER_MIN)

        advertiser.deposit_attractive(pher_attr, target[0], target[1], ROBOT_RADIUS)
        try:
            advertiser.x, advertiser.y = next(spiral_iter)
        except StopIteration:
            pass

        for searcher in searchers:
            if searcher.mode == "FOUND":
                path_rows.append(
                    _path_row(
                        grid_size=grid_size,
                        n_search_robots=n_search_robots,
                        requested_distance_shell=requested_distance_shell,
                        spawn_position_id=spawn_position_id,
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
            if searcher.senses_attractive(pher_attr, ROBOT_RADIUS) and detection_step_by_id[searcher.id] is None:
                detection_step_by_id[searcher.id] = step
                detection_distance_by_id[searcher.id] = current_distance

            mark_visible(searcher.local_covered, searcher.x, searcher.y, ROBOT_RADIUS)
            mark_visible(covered, searcher.x, searcher.y, ROBOT_RADIUS)
            discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, ROBOT_RADIUS)

            if target in found_targets:
                searcher.mode = "FOUND"
            else:
                searcher.step_search_or_follow(
                    pher_rep=pher_rep,
                    pher_attr=pher_attr,
                    targets=targets,
                    all_robots=robots,
                    global_target_visits=global_target_visits,
                    robot_radius=ROBOT_RADIUS,
                    collision_radius=COLLISION_RADIUS,
                    pher_deposit=PHER_DEPOSIT,
                    attractive_sensing_radius=ROBOT_RADIUS,
                )

            if previous_mode == "SEARCH" and searcher.mode == "FOLLOW" and mode_switch_step_by_id[searcher.id] is None:
                mode_switch_step_by_id[searcher.id] = step

            mark_visible(searcher.local_covered, searcher.x, searcher.y, ROBOT_RADIUS)
            mark_visible(covered, searcher.x, searcher.y, ROBOT_RADIUS)
            discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, ROBOT_RADIUS)

            new_distance = manhattan((searcher.x, searcher.y), target)
            final_distance_by_id[searcher.id] = new_distance
            if new_distance >= previous_distances[searcher.id]:
                stagnation_count_by_id[searcher.id] += 1
                if detection_step_by_id[searcher.id] is not None:
                    post_detection_stagnation_count_by_id[searcher.id] += 1
            previous_distances[searcher.id] = new_distance

            path_rows.append(
                _path_row(
                    grid_size=grid_size,
                    n_search_robots=n_search_robots,
                    requested_distance_shell=requested_distance_shell,
                    spawn_position_id=spawn_position_id,
                    algorithm_seed=algorithm_seed,
                    searcher=searcher,
                    target=target,
                    step=step,
                    attractive_detected=detection_step_by_id[searcher.id] is not None,
                )
            )

            if searcher.mode == "FOUND" or target in found_targets:
                found = True
                found_robot_id = searcher.id
                found_step_by_id[searcher.id] = step
                steps_to_target = step
                break

        swarm_rows.append(
            _swarm_row(
                grid_size=grid_size,
                n_search_robots=n_search_robots,
                requested_distance_shell=requested_distance_shell,
                spawn_position_id=spawn_position_id,
                algorithm_seed=algorithm_seed,
                searchers=searchers,
                target=target,
                step=step,
                found=found,
            )
        )
        if found:
            break

    optimal_manhattan_steps = min(initial_distances.values())
    found_detection_step = detection_step_by_id.get(found_robot_id) if found else None
    found_detection_distance = detection_distance_by_id.get(found_robot_id, np.nan) if found else np.nan
    time_to_attractive_detection = found_detection_step if found_detection_step is not None else np.nan
    time_from_detection_to_target = (
        steps_to_target - found_detection_step if found and found_detection_step is not None else np.nan
    )
    path_efficiency = optimal_manhattan_steps / steps_to_target if found and steps_to_target > 0 else np.nan
    post_detection_path_efficiency = (
        found_detection_distance / time_from_detection_to_target
        if found and time_from_detection_to_target and time_from_detection_to_target > 0
        else np.nan
    )
    convergence_rate = (
        (found_detection_distance - final_distance_by_id[int(found_robot_id)]) / time_from_detection_to_target
        if found and found_detection_step is not None and time_from_detection_to_target > 0
        else np.nan
    )

    robot_rows = []
    for searcher in searchers:
        robot_found_step = found_step_by_id[searcher.id]
        robot_detection_step = detection_step_by_id[searcher.id]
        robot_time_from_detection = (
            robot_found_step - robot_detection_step
            if robot_found_step is not None and robot_detection_step is not None
            else np.nan
        )
        robot_path_efficiency = (
            initial_distances[searcher.id] / robot_found_step
            if robot_found_step is not None and robot_found_step > 0
            else np.nan
        )
        robot_post_detection_efficiency = (
            detection_distance_by_id[searcher.id] / robot_time_from_detection
            if robot_time_from_detection and robot_time_from_detection > 0
            else np.nan
        )
        robot_convergence_rate = (
            (detection_distance_by_id[searcher.id] - final_distance_by_id[searcher.id]) / robot_time_from_detection
            if robot_detection_step is not None and robot_time_from_detection and robot_time_from_detection > 0
            else np.nan
        )
        robot_rows.append(
            {
                "grid_size": grid_size,
                "n_search_robots": n_search_robots,
                "requested_distance_shell": requested_distance_shell,
                "spawn_position_id": spawn_position_id,
                "algorithm_seed": algorithm_seed,
                "target_x": target[0],
                "target_y": target[1],
                "search_robot_id": searcher.id,
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
                "path_efficiency": robot_path_efficiency,
                "post_detection_path_efficiency": robot_post_detection_efficiency,
                "convergence_rate": robot_convergence_rate,
                "stagnation_count": stagnation_count_by_id[searcher.id],
                "post_detection_stagnation_count": post_detection_stagnation_count_by_id[searcher.id],
                "final_distance": final_distance_by_id[searcher.id],
            }
        )

    return {
        "grid_size": grid_size,
        "n_search_robots": n_search_robots,
        "target_x": target[0],
        "target_y": target[1],
        "requested_distance_shell": requested_distance_shell,
        "spawn_position_id": spawn_position_id,
        "search_starts": ";".join(f"{x},{y}" for x, y in search_starts),
        "search_start_distances": ";".join(str(initial_distances[searcher.id]) for searcher in searchers),
        "actual_min_start_distance": min(initial_distances.values()),
        "actual_mean_start_distance": float(np.mean(list(initial_distances.values()))),
        "algorithm_seed": algorithm_seed,
        "max_horizon": max_horizon,
        "found": int(found),
        "found_robot_id": found_robot_id,
        "steps_to_target": steps_to_target,
        "optimal_manhattan_steps": optimal_manhattan_steps,
        "excess_steps_over_optimal": steps_to_target - optimal_manhattan_steps if found else np.nan,
        "slowdown_ratio": steps_to_target / optimal_manhattan_steps if found and optimal_manhattan_steps > 0 else np.nan,
        "path_efficiency": path_efficiency,
        "time_to_attractive_detection": time_to_attractive_detection,
        "time_from_detection_to_target": time_from_detection_to_target,
        "post_detection_path_efficiency": post_detection_path_efficiency,
        "convergence_rate": convergence_rate,
        "stagnation_count": stagnation_count_by_id.get(found_robot_id, np.nan) if found else np.nan,
        "mode_switch_step": mode_switch_step_by_id.get(found_robot_id, np.nan) if found else np.nan,
        "_path_rows": path_rows,
        "_swarm_rows": swarm_rows,
        "_robot_rows": robot_rows,
    }


Task = Tuple[int, int, int, int, int, List[Tuple[int, int]], int]


def _build_tasks() -> List[Task]:
    tasks: List[Task] = []
    order = 0
    for grid_size in GRID_SIZES:
        for n_search_robots in SEARCH_ROBOT_COUNTS:
            spawn_sets = generate_spawn_position_sets(grid_size, DISTANCE_SHELLS, n_search_robots)
            for spawn_position_id, (requested_distance_shell, search_starts) in enumerate(spawn_sets, 1):
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
                            search_starts,
                            algorithm_seed,
                        )
                    )
                    order += 1
    return tasks


def _run_task(task: Task) -> Tuple[int, Dict[str, object]]:
    order, grid_size, n_search_robots, requested_distance_shell, spawn_position_id, search_starts, algorithm_seed = task
    print(
        f"Running rendezvous: grid={grid_size}, searchers={n_search_robots}, "
        f"shell={requested_distance_shell}, starts={search_starts}, seed={algorithm_seed}",
        flush=True,
    )
    return order, run_simulation(
        grid_size,
        n_search_robots,
        requested_distance_shell,
        spawn_position_id,
        search_starts,
        algorithm_seed,
    )


def _write_results(rows: List[Dict[str, object]]) -> Path:
    output_dir = Path.cwd() / "experiment_results_rendezvous"
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
        df.groupby(["grid_size", "n_search_robots", "requested_distance_shell"], as_index=False)
        .agg(
            total_runs=("steps_to_target", "size"),
            success_rate_by_distance=("found", "mean"),
            avg_steps_to_target=("steps_to_target", "mean"),
            std_steps_to_target=("steps_to_target", "std"),
            avg_optimal_manhattan_steps=("optimal_manhattan_steps", "mean"),
            avg_excess_steps=("excess_steps_over_optimal", "mean"),
            avg_slowdown_ratio=("slowdown_ratio", "mean"),
            avg_path_efficiency=("path_efficiency", "mean"),
            avg_time_to_attractive_detection=("time_to_attractive_detection", "mean"),
            avg_time_from_detection_to_target=("time_from_detection_to_target", "mean"),
            avg_post_detection_path_efficiency=("post_detection_path_efficiency", "mean"),
            avg_convergence_rate=("convergence_rate", "mean"),
            avg_stagnation_count=("stagnation_count", "mean"),
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


def run_experiments(workers: Optional[int] = None) -> None:
    tasks = _build_tasks()
    max_workers = workers or os.cpu_count() or 1
    completed: Dict[int, Dict[str, object]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in as_completed(futures):
            order, result = future.result()
            completed[order] = result
            print(
                f"Completed {len(completed)}/{len(tasks)}: "
                f"found={result['found']}, steps={result['steps_to_target']}, "
                f"shell={result['requested_distance_shell']}, searchers={result['n_search_robots']}",
                flush=True,
            )

    rows = [completed[i] for i in range(len(tasks))]
    xlsx_path = _write_results(rows)
    print(f"\nResults saved to: {xlsx_path}")
    print("Additional CSVs saved beside the workbook: search_robot_paths.csv, swarm_convergence.csv, search_robot_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Defaults to CPU count.")
    args = parser.parse_args()
    run_experiments(workers=args.workers)
