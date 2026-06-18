import argparse
from pathlib import Path
from typing import Dict, Set, Tuple
import sys

import numpy as np
from matplotlib.patches import Circle

from robot_random import Robot
from robotarium_swarm_common import *

ROBOTARIUM_RANDOM_WALK_GRID_STEP_CELLS = ROBOT_RADIUS + 2
ROBOTARIUM_RANDOM_WALK_ARRIVAL_CELLS = 1.25
ROBOTARIUM_RANDOM_WALK_COLLISION_CELLS = max(COLLISION_RADIUS, ROBOT_RADIUS)


def clamp_robotarium_goal(goal: Tuple[int, int]) -> Tuple[int, int]:
    margin = ROBOT_RADIUS
    gx, gy = goal
    return (
        int(np.clip(gx, margin, W - 1 - margin)),
        int(np.clip(gy, margin, H - 1 - margin)),
    )


def create_random_walk_motion_helpers():
    si_barrier_cert = create_si_barrier_certificate_with_boundary(
        safety_radius=ROBOTARIUM_CONTROL_SAFETY_RADIUS,
        barrier_gain=60.0,
        magnitude_limit=ROBOTARIUM_CONTROL_MAGNITUDE_LIMIT,
        boundary_points=ACTIVE_WORLD_BOUNDS,
    )
    static_barrier_cert = create_si_barrier_certificate_with_boundary(
        safety_radius=ROBOTARIUM_CONTROL_SAFETY_RADIUS,
        barrier_gain=100.0,
        magnitude_limit=ROBOTARIUM_CONTROL_MAGNITUDE_LIMIT,
        boundary_points=ACTIVE_WORLD_BOUNDS,
    )
    si_position_controller = create_si_position_controller(
        x_velocity_gain=0.8,
        y_velocity_gain=0.8,
        velocity_magnitude_limit=ROBOTARIUM_CONTROL_MAGNITUDE_LIMIT,
    )
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(
        projection_distance=SI_PROJECTION_DISTANCE,
    )
    return si_barrier_cert, static_barrier_cert, si_position_controller, si_to_uni_dyn, uni_to_si_states


def covered_to_rgba(covered: np.ndarray) -> np.ndarray:
    rgba = np.zeros((covered.shape[0], covered.shape[1], 4), dtype=float)
    covered_cells = covered > 0
    rgba[covered_cells, 0] = 0.00
    rgba[covered_cells, 1] = 0.00
    rgba[covered_cells, 2] = 1.00
    rgba[covered_cells, 3] = 0.58
    return rgba


def add_coverage_overlay(r, covered: np.ndarray, show_figure: bool):
    if not show_figure:
        return None, None
    ax = get_robotarium_axes(r)
    if ax is None:
        return None, None
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


def add_safety_area_plots(r, show_figure: bool, show_safe_area: bool):
    if not show_figure or not show_safe_area:
        return None
    ax = get_robotarium_axes(r)
    if ax is None:
        return None
    safety_plots = []
    for _ in range(N_ROBOTS):
        circle = Circle(
            (0.0, 0.0),
            ROBOTARIUM_SAFETY_RADIUS,
            fill=False,
            edgecolor="tab:cyan",
            linewidth=1.2,
            linestyle="--",
            alpha=0.45,
            zorder=18,
        )
        ax.add_patch(circle)
        safety_plots.append(circle)
    return safety_plots


def refresh_safety_area_plots(safety_plots, x_si: np.ndarray,
                              failed_robot_ids: Set[int]) -> None:
    if safety_plots is None:
        return
    for robot_id, circle in enumerate(safety_plots):
        circle.center = (float(x_si[0, robot_id]), float(x_si[1, robot_id]))
        circle.set_radius(ROBOTARIUM_SAFETY_RADIUS)
        if robot_id in failed_robot_ids:
            circle.set_edgecolor("red")
            circle.set_linestyle("-")
            circle.set_linewidth(2.2)
            circle.set_alpha(0.95)
            circle.set_zorder(24)
        else:
            circle.set_edgecolor("tab:cyan")
            circle.set_linestyle("--")
            circle.set_linewidth(1.2)
            circle.set_alpha(0.45)
            circle.set_zorder(18)


def run(show_figure: bool = True, max_steps: int = MAX_STEPS,
        grid_step_size: int = ROBOTARIUM_RANDOM_WALK_GRID_STEP_CELLS,
        log_interval: int = 25,
        waypoint_update_steps: int = WAYPOINT_UPDATE_STEPS,
        show_grid: bool = True,
        show_safe_area: bool = False) -> Dict[str, object]:
    robot_rng = make_algorithm_rng()
    start_wall_time = wall_time()
    r = create_robotarium(
        show_figure,
        show_grid,
        initial_grid_positions=SHARED_INITIAL_GRID_POSITIONS,
        seed=START_RANDOM_SEED,
    )
    helpers = create_random_walk_motion_helpers()
    x_pose = r.get_poses()
    initial_pose = x_pose.copy()
    _, _, _, _, uni_to_si_states = helpers
    x_si = uni_to_si_states(x_pose)

    robots = [
        Robot(
            id=i,
            x=world_to_grid(x_si[0, i], x_si[1, i])[0],
            y=world_to_grid(x_si[0, i], x_si[1, i])[1],
            robot_radius=ROBOT_RADIUS,
            collision_radius=ROBOTARIUM_RANDOM_WALK_COLLISION_CELLS,
        )
        for i in range(N_ROBOTS)
    ]
    targets = canonical_targets()
    print_seed_configuration("Random-walk", targets, SHARED_INITIAL_GRID_POSITIONS)
    failure_schedule = make_e2_failure_schedule(max_horizon=max_steps)
    print_failure_configuration("Random-walk", failure_schedule)
    failed_robot_ids: Set[int] = set()
    found_targets: Set[Tuple[int, int]] = set()
    covered = np.zeros((H, W), dtype=bool)
    local_maps = [np.zeros((H, W), dtype=bool) for _ in range(N_ROBOTS)]
    grid_goals = [clamp_robotarium_goal((robot.x, robot.y)) for robot in robots]
    stuck_counts = np.zeros(N_ROBOTS, dtype=int)
    moving_robot_id = None
    moving_goal_active = False
    mover_slot = 0
    coverage_plot, coverage_label = add_coverage_overlay(r, covered, show_figure)
    found_plot = add_target_plot(r, targets, found_targets, "Memoryless Random Walk Robotarium", 0, show_figure)
    safety_plots = add_safety_area_plots(r, show_figure, show_safe_area)
    refresh_safety_area_plots(safety_plots, x_si, failed_robot_ids)

    step = 0
    while len(found_targets) < len(targets) and not should_stop_simulation(step, max_steps, start_wall_time):
        update_failed_robots(step, failure_schedule, failed_robot_ids, "Random-walk")
        active_indices = active_robot_indices(failed_robot_ids)
        x_si = uni_to_si_states(x_pose)
        refresh_safety_area_plots(safety_plots, x_si, failed_robot_ids)
        positions = [world_to_grid(x_si[0, i], x_si[1, i]) for i in range(N_ROBOTS)]
        for i in active_indices:
            robot = robots[i]
            robot.x, robot.y = positions[i]

        if moving_robot_id not in active_indices:
            moving_goal_active = False

        active_positions = [positions[i] for i in active_indices]
        active_maps = [local_maps[i] for i in active_indices]
        sense_targets(active_positions, active_maps, covered, targets, found_targets)
        refresh_coverage_overlay(coverage_plot, coverage_label, covered)
        for i in active_indices:
            if i != moving_robot_id:
                grid_goals[i] = positions[i]

        if active_indices and not moving_goal_active:
            moving_robot_id = active_indices[mover_slot % len(active_indices)]
            mover_slot += 1
            robot = robots[moving_robot_id]
            start = positions[moving_robot_id]
            waypoint = robot.choose_move(robot_rng, W, H, robots, heading=x_pose[2, moving_robot_id])
            grid_goals[moving_robot_id] = clamp_robotarium_goal(scale_grid_move(start, waypoint, grid_step_size))
            robot.x, robot.y = grid_goals[moving_robot_id]
            moving_goal_active = True

        if (
            moving_robot_id in active_indices
            and has_arrived(
                x_si,
                moving_robot_id,
                grid_goals[moving_robot_id],
                tolerance_cells=ROBOTARIUM_RANDOM_WALK_ARRIVAL_CELLS,
            )
        ):
            moving_goal_active = False

        goals = make_goal_world(grid_goals)
        goals = hold_failed_robot_goals(goals, x_si, failed_robot_ids)
        prev_x_si = x_si.copy()
        x_pose = update_robotarium(r, x_pose, goals, helpers, failed_robot_ids=failed_robot_ids)
        moved_x_si = uni_to_si_states(x_pose)
        refresh_safety_area_plots(safety_plots, moved_x_si, failed_robot_ids)
        step += 1
        movement = np.linalg.norm(moved_x_si - prev_x_si, axis=0)
        moved_positions = [world_to_grid(moved_x_si[0, i], moved_x_si[1, i]) for i in range(N_ROBOTS)]
        for i in active_indices:
            if movement[i] < STUCK_MOVEMENT_EPS:
                stuck_counts[i] += 1
            else:
                stuck_counts[i] = 0
            if i == moving_robot_id and stuck_counts[i] >= STUCK_STEPS:
                robot = robots[i]
                start = moved_positions[i]
                waypoint = robot.choose_move(robot_rng, W, H, robots, heading=x_pose[2, i])
                grid_goals[i] = clamp_robotarium_goal(scale_grid_move(start, waypoint, grid_step_size))
                robot.x, robot.y = grid_goals[i]
                stuck_counts[i] = 0
            elif stuck_counts[i] >= STUCK_STEPS:
                stuck_counts[i] = 0
        print_movement_report("Random-walk", step, prev_x_si, moved_x_si, log_interval)
        refresh_found_plot(r, found_plot, found_targets, "Memoryless Random Walk Robotarium", step, len(targets))

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
        f"Memoryless random walk Robotarium completed: found {len(found_targets)}/{len(targets)} "
        f"targets in {step} steps. Wall time in {wall_time() - start_wall_time} sec."
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
    parser.add_argument("--grid-step-size", type=int, default=ROBOTARIUM_RANDOM_WALK_GRID_STEP_CELLS)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--waypoint-update-steps", type=int, default=WAYPOINT_UPDATE_STEPS)
    parser.add_argument("--show-grid", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument(
        "--show-safe-area",
        nargs="?",
        const=True,
        default=False,
        type=lambda value: value if isinstance(value, bool) else value.lower() == "true",
    )
    args = parser.parse_args()
    run(
        show_figure=not args.no_show,
        max_steps=args.max_steps,
        grid_step_size=args.grid_step_size,
        log_interval=args.log_interval,
        waypoint_update_steps=args.waypoint_update_steps,
        show_grid=args.show_grid,
        show_safe_area=args.show_safe_area,
    )
