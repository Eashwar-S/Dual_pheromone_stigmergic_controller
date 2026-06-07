import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

import numpy as np

# SIM_ROOT = Path(__file__).resolve().parent
# PROJECT_ROOT = SIM_ROOT.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from lawnmower_pattern import generate_sweep_path_for_region
from partitioning import lloyd_balanced
from robot import Robot
from robotarium_swarm_common import *

PATH_SMOOTHING_PASSES = 2
WAYPOINT_SPACING_CELLS = 2.5
WAYPOINT_ARRIVAL_CELLS = 1.3
SIDE_BUFFER_CELLS = 1
YIELD_DISTANCE = 0.22
MIN_GOAL_SEPARATION = 0.16
YIELD_PERIOD_STEPS = 50


def _path_to_world_arrays(path: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
    coords = [grid_to_world(gx, gy) for gx, gy in path]
    if not coords:
        return [], []
    xs, ys = zip(*coords)
    return list(xs), list(ys)


def _world_path_to_arrays(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if path.size == 0:
        return np.array([]), np.array([])
    return path[:, 0], path[:, 1]


def _dedupe_world_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-9:
            keep.append(i)
    return points[keep]


def _chaikin_path(points: np.ndarray, passes: int) -> np.ndarray:
    smoothed = _dedupe_world_points(points)
    for _ in range(passes):
        if len(smoothed) < 3:
            break
        next_points = [smoothed[0]]
        for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
            next_points.append(0.75 * p0 + 0.25 * p1)
            next_points.append(0.25 * p0 + 0.75 * p1)
        next_points.append(smoothed[-1])
        smoothed = _dedupe_world_points(np.array(next_points))
    return smoothed


def _resample_world_path(points: np.ndarray, spacing: float) -> np.ndarray:
    points = _dedupe_world_points(points)
    if len(points) <= 1:
        return points

    resampled = [points[0]]
    carried = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        segment = p1 - p0
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        direction = segment / length
        distance = spacing - carried
        while distance < length:
            resampled.append(p0 + direction * distance)
            distance += spacing
        carried = max(0.0, length - (distance - spacing))
        if carried <= 1e-9:
            carried = 0.0
    if np.linalg.norm(resampled[-1] - points[-1]) > 1e-9:
        resampled.append(points[-1])
    return np.array(resampled)


def make_unicycle_waypoint_path(path: List[Tuple[int, int]]) -> np.ndarray:
    points = np.array([grid_to_world(gx, gy) for gx, gy in path], dtype=float)
    if len(points) <= 2:
        return points
    smoothed = _chaikin_path(points, PATH_SMOOTHING_PASSES)
    spacing = WAYPOINT_SPACING_CELLS * min(CELL_WIDTH, CELL_HEIGHT)
    return _resample_world_path(smoothed, spacing)


def add_planned_path_plots(r, waypoint_paths: List[np.ndarray], show_figure: bool):
    if not show_figure:
        return None
    ax = r._axes_handle
    followed_lines = []
    for waypoint_path in waypoint_paths:
        xs, ys = _world_path_to_arrays(waypoint_path)
        ax.plot(xs, ys, color="0.68", linewidth=1.1, alpha=0.7, zorder=-1)
        followed_line, = ax.plot([], [], color="0.20", linewidth=1.8, alpha=0.95, zorder=1)
        followed_lines.append(followed_line)
    return followed_lines


def refresh_followed_path_plots(path_plots, waypoint_paths: List[np.ndarray],
                                waypoint_indices: List[int]) -> None:
    if path_plots is None:
        return
    for line, waypoint_path, waypoint_idx in zip(path_plots, waypoint_paths, waypoint_indices):
        xs, ys = _world_path_to_arrays(waypoint_path[:waypoint_idx + 1])
        line.set_data(xs, ys)


def covered_to_rgba(covered: np.ndarray) -> np.ndarray:
    rgba = np.zeros((covered.shape[0], covered.shape[1], 4), dtype=float)
    covered_cells = covered > 0
    rgba[covered_cells, 0] = 0.18
    rgba[covered_cells, 1] = 0.18
    rgba[covered_cells, 2] = 0.18
    rgba[covered_cells, 3] = 0.58
    return rgba


def add_coverage_overlay(r, covered: np.ndarray, show_figure: bool):
    if not show_figure:
        return None, None
    ax = r._axes_handle
    coverage_plot = ax.imshow(
        covered_to_rgba(covered),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        zorder=-2,
        interpolation="nearest",
    )
    coverage_label = ax.text(
        0.02,
        0.98,
        "Covered cells: 0",
        color="0.12",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax.transAxes,
        zorder=30,
        bbox={"facecolor": "white", "edgecolor": "0.25", "alpha": 0.8, "pad": 4},
    )
    return coverage_plot, coverage_label


def refresh_coverage_overlay(coverage_plot, coverage_label, covered: np.ndarray) -> None:
    if coverage_plot is None:
        return
    coverage_plot.set_data(covered_to_rgba(covered))
    if coverage_label is not None:
        coverage_label.set_text(f"Covered cells: {int(np.count_nonzero(covered))}")


def create_centralized_motion_helpers():
    si_barrier_cert = create_si_barrier_certificate_with_boundary(
        safety_radius=0.13,
        barrier_gain=60.0,
        magnitude_limit=0.16,
        boundary_points=ACTIVE_WORLD_BOUNDS,
    )
    si_position_controller = create_si_position_controller(
        x_velocity_gain=0.8,
        y_velocity_gain=0.8,
        velocity_magnitude_limit=0.14,
    )
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(
        projection_distance=SI_PROJECTION_DISTANCE,
    )
    return si_barrier_cert, si_position_controller, si_to_uni_dyn, uni_to_si_states


def make_initial_conditions_from_grid(grid_positions: List[Tuple[int, int]],
                                      seed: int = START_RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = np.zeros((3, N_ROBOTS), dtype=float)
    for i, (gx, gy) in enumerate(grid_positions[:N_ROBOTS]):
        poses[0, i], poses[1, i] = grid_to_world(gx, gy)
    poses[2, :] = rng.uniform(-np.pi, np.pi, N_ROBOTS)
    return poses


def create_centralized_robotarium(show_figure: bool, show_grid: bool,
                                  initial_grid_positions: List[Tuple[int, int]]):
    r = robotarium.Robotarium(
        number_of_robots=N_ROBOTS,
        show_figure=show_figure,
        sim_in_real_time=False,
        initial_conditions=make_initial_conditions_from_grid(initial_grid_positions),
        skip_initialization=True,
    )
    if show_figure:
        ax = r._axes_handle
        ax.set_xlim(WORLD_BOUNDS[0], WORLD_BOUNDS[1])
        ax.set_ylim(WORLD_BOUNDS[2], WORLD_BOUNDS[3])
        ax.set_aspect("equal")
        if show_grid:
            x_lines = np.linspace(ACTIVE_WORLD_BOUNDS[0], ACTIVE_WORLD_BOUNDS[1], W + 1)
            y_lines = np.linspace(ACTIVE_WORLD_BOUNDS[2], ACTIVE_WORLD_BOUNDS[3], H + 1)
            ax.vlines(x_lines, ACTIVE_WORLD_BOUNDS[2], ACTIVE_WORLD_BOUNDS[3],
                      color="k", alpha=0.18, linewidth=0.35, zorder=-3)
            ax.hlines(y_lines, ACTIVE_WORLD_BOUNDS[0], ACTIVE_WORLD_BOUNDS[1],
                      color="k", alpha=0.18, linewidth=0.35, zorder=-3)
    return r


def advance_waypoints(x_si: np.ndarray, waypoint_paths: List[np.ndarray],
                      waypoint_indices: List[int]) -> None:
    tolerance = WAYPOINT_ARRIVAL_CELLS * min(CELL_WIDTH, CELL_HEIGHT)
    for i, waypoint_path in enumerate(waypoint_paths):
        while waypoint_indices[i] < len(waypoint_path) - 1:
            current_goal = waypoint_path[waypoint_indices[i]]
            if np.linalg.norm(x_si[:, i] - current_goal) > tolerance:
                break
            waypoint_indices[i] += 1


def make_waypoint_goals(waypoint_paths: List[np.ndarray],
                        waypoint_indices: List[int]) -> np.ndarray:
    goals = np.zeros((2, N_ROBOTS), dtype=float)
    for i, (waypoint_path, waypoint_idx) in enumerate(zip(waypoint_paths, waypoint_indices)):
        goals[:, i] = waypoint_path[waypoint_idx]
    return goals


def apply_deadlock_yield(goals: np.ndarray, x_si: np.ndarray, step: int) -> np.ndarray:
    adjusted = goals.copy()
    for i in range(N_ROBOTS - 1):
        for j in range(i + 1, N_ROBOTS):
            separation = x_si[:, i] - x_si[:, j]
            distance = float(np.linalg.norm(separation))
            if distance <= 1e-9:
                closing = True
            else:
                desired_i = goals[:, i] - x_si[:, i]
                desired_j = goals[:, j] - x_si[:, j]
                closing = float(np.dot(desired_i - desired_j, separation)) < 0.0
            goals_too_close = np.linalg.norm(goals[:, i] - goals[:, j]) < MIN_GOAL_SEPARATION
            if distance < YIELD_DISTANCE and (closing or goals_too_close):
                holder = i if ((step // YIELD_PERIOD_STEPS + i + j) % 2) else j
                adjusted[:, holder] = x_si[:, holder]
    return adjusted


def border_safe_mask(buffer_cells: int) -> np.ndarray:
    margin = max(0, buffer_cells)
    safe = np.zeros((H, W), dtype=bool)
    if margin * 2 >= min(H, W):
        safe[:, :] = True
        return safe
    safe[margin:H - margin, margin:W - margin] = True
    return safe


def nearest_safe_cell(point: Tuple[float, float], allowed: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(allowed)
    if xs.size == 0:
        return (
            int(np.clip(round(point[0] - 0.5), 0, W - 1)),
            int(np.clip(round(point[1] - 0.5), 0, H - 1)),
        )
    d = (xs + 0.5 - point[0]) ** 2 + (ys + 0.5 - point[1]) ** 2
    k = int(np.argmin(d))
    return int(xs[k]), int(ys[k])


def build_robots(rng: np.random.Generator) -> Tuple[List[Robot], np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    labels, centers = lloyd_balanced(points, N_ROBOTS, 10, 30, 0.1, 0.1, rng)
    zones = labels.reshape(H, W)
    safe_border = border_safe_mask(SIDE_BUFFER_CELLS)

    paths: List[List[Tuple[int, int]]] = []
    for i in range(N_ROBOTS):
        region = zones == i
        sweep_region = region & safe_border
        sweep = generate_sweep_path_for_region(sweep_region, robot_radius=ROBOT_RADIUS)
        if not sweep:
            cx, cy = centers[i]
            fallback_region = safe_border if safe_border.any() else region
            sweep = [nearest_safe_cell((cx, cy), fallback_region)]
        paths.append(sweep)
    return [Robot(i, paths[i]) for i in range(N_ROBOTS)], zones


def run(show_figure: bool = True, max_steps: int = MAX_STEPS,
        grid_step_size: int = GRID_MOVE_CELLS,
        show_grid: bool = True) -> Dict[str, object]:
    planner_rng = np.random.default_rng(START_RANDOM_SEED)
    start_wall_time = wall_time()
    robots, zones = build_robots(planner_rng)
    initial_grid_positions = [robot.path[0] for robot in robots]
    r = create_centralized_robotarium(show_figure, show_grid, initial_grid_positions)
    helpers = create_centralized_motion_helpers()
    x_pose = r.get_poses()
    initial_pose = x_pose.copy()
    _, _, _, uni_to_si_states = helpers
    waypoint_paths = [make_unicycle_waypoint_path(robot.path) for robot in robots]
    waypoint_indices = [0 for _ in waypoint_paths]
    add_partition_background(r, zones, show_figure)
    path_plots = add_planned_path_plots(r, waypoint_paths, show_figure)
    refresh_followed_path_plots(path_plots, waypoint_paths, waypoint_indices)
    targets = canonical_targets()
    # print(f'targets - {targets}')
    print_seed_configuration("Centralized", targets, initial_grid_positions)
    failure_schedule = make_e2_failure_schedule(max_horizon=max_steps)
    # print(failure_schedule)
    print_failure_configuration("Centralized", failure_schedule)
    failed_robot_ids: Set[int] = set()
    found_targets: Set[Tuple[int, int]] = set()
    covered = np.zeros((H, W), dtype=int)
    local_maps = [np.zeros((H, W), dtype=bool) for _ in range(N_ROBOTS)]
    coverage_plot, coverage_label = add_coverage_overlay(r, covered, show_figure)
    found_plot = add_target_plot(r, targets, found_targets, "Centralized Robotarium", 0, show_figure)

    step = 0
    while step < max_steps and len(found_targets) < len(targets):
        update_failed_robots(step, failure_schedule, failed_robot_ids, "Centralized")
        active_indices = active_robot_indices(failed_robot_ids)
        x_si = uni_to_si_states(x_pose)
        physical_positions = [world_to_grid(x_si[0, i], x_si[1, i]) for i in range(N_ROBOTS)]
        active_positions = [physical_positions[i] for i in active_indices]
        active_maps = [local_maps[i] for i in active_indices]
        sense_targets(active_positions, active_maps, covered, targets, found_targets)
        refresh_coverage_overlay(coverage_plot, coverage_label, covered)
        advance_waypoints(x_si, waypoint_paths, waypoint_indices)
        goals = make_waypoint_goals(waypoint_paths, waypoint_indices)
        goals = apply_deadlock_yield(goals, x_si, step)
        goals = hold_failed_robot_goals(goals, x_si, failed_robot_ids)
        x_pose = update_robotarium(r, x_pose, goals, helpers, failed_robot_ids=failed_robot_ids)
        step += 1
        refresh_followed_path_plots(path_plots, waypoint_paths, waypoint_indices)
        refresh_found_plot(r, found_plot, found_targets, "Centralized Robotarium", step, len(targets))

    if len(found_targets) >= len(targets):
        add_completion_banner(
            r,
            target_step=step,
            sim_seconds=step * r.TIME_STEP,
            wall_seconds=wall_time() - start_wall_time,
            show_figure=show_figure,
        )

    displacement = np.linalg.norm(x_pose[:2, :] - initial_pose[:2, :], axis=0)
    print(
        f"Centralized Robotarium completed: found {len(found_targets)}/{len(targets)} "
        f"targets in {step} steps. Wall time is {wall_time() - start_wall_time} sec"
    )
    finish_robotarium(r)
    return {
        "steps": step,
        "targets_found": len(found_targets),
        "n_targets": len(targets),
        "max_displacement": float(displacement.max()),
        "failure_schedule": failure_schedule,
        "failed_robot_ids": sorted(failed_robot_ids),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--grid-step-size", type=int, default=GRID_MOVE_CELLS)
    parser.add_argument("--show-grid", type=lambda value: value.lower() == "true", default=True)
    args = parser.parse_args()
    run(
        show_figure=not args.no_show,
        max_steps=args.max_steps,
        grid_step_size=args.grid_step_size,
        show_grid=args.show_grid,
    )
