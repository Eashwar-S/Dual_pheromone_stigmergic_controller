import argparse
import random
from pathlib import Path
from typing import Dict, Set, Tuple
import sys

import numpy as np

from robotarium_swarm_common import *
from robot_efficient import Robot

def apply_side_buffer(goal: Tuple[int, int]) -> Tuple[int, int]:
    gx, gy = goal
    margin = ROBOT_RADIUS
    return (
        int(np.clip(gx, margin, W - 1 - margin)),
        int(np.clip(gy, margin, H - 1 - margin)),
    )


def covered_to_rgba(covered: np.ndarray) -> np.ndarray:
    rgba = np.zeros((covered.shape[0], covered.shape[1], 4), dtype=float)
    covered_cells = covered > 0
    rgba[covered_cells, 0] = 0.00
    rgba[covered_cells, 1] = 0.00
    rgba[covered_cells, 2] = 1.00
    rgba[covered_cells, 3] = 0.88
    return rgba


def add_coverage_overlay(r, covered: np.ndarray, show_figure: bool):
    if not show_figure:
        return None
    ax = get_robotarium_axes(r)
    if ax is None:
        return None
    coverage_plot = ax.imshow(
        covered_to_rgba(covered),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        zorder=-5,
        interpolation="nearest",
    )
    return coverage_plot


def refresh_coverage_overlay(coverage_plot, covered: np.ndarray) -> None:
    if coverage_plot is not None:
        coverage_plot.set_data(covered_to_rgba(covered))


def run(show_figure: bool = True, max_steps: int = MAX_STEPS,
        grid_step_size: int = GRID_MOVE_CELLS,
        log_interval: int = 25,
        waypoint_update_steps: int = WAYPOINT_UPDATE_STEPS,
        show_grid: bool = True) -> Dict[str, object]:
    random.seed(ALGORITHM_RANDOM_SEED)
    start_wall_time = wall_time()
    r = create_robotarium(
        show_figure,
        show_grid,
        initial_grid_positions=SHARED_INITIAL_GRID_POSITIONS,
        seed=START_RANDOM_SEED,
    )
    helpers = create_motion_helpers()
    x_pose = r.get_poses()
    initial_pose = x_pose.copy()
    _, _, _, uni_to_si_states = helpers
    x_si = uni_to_si_states(x_pose)

    robots = []
    for i in range(N_ROBOTS):
        gx, gy = world_to_grid(x_si[0, i], x_si[1, i])
        robots.append(
            Robot(
                id=i,
                x=gx,
                y=gy,
                local_covered=np.zeros((H, W), dtype=bool),
            )
        )

    targets = canonical_targets()
    print_seed_configuration("Search", targets, SHARED_INITIAL_GRID_POSITIONS)
    failure_schedule = make_e2_failure_schedule(max_horizon=max_steps)
    print_failure_configuration("Search", failure_schedule)
    failed_robot_ids: Set[int] = set()
    found_targets: Set[Tuple[int, int]] = set()
    covered = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)
    grid_goals = [apply_side_buffer((robot.x, robot.y)) for robot in robots]
    waypoint_timers = np.zeros(N_ROBOTS, dtype=int)
    stuck_counts = np.zeros(N_ROBOTS, dtype=int)
    coverage_plot = add_coverage_overlay(r, covered, show_figure)
    pheromone_plot = add_pheromone_overlay(r, pher, show_figure)
    found_plot = add_target_plot(r, targets, found_targets, "Search Stigmergy Robotarium", 0, show_figure)

    step = 0
    while len(found_targets) < len(targets) and not should_stop_simulation(step, max_steps, start_wall_time):
        update_failed_robots(step, failure_schedule, failed_robot_ids, "Search")
        active_indices = active_robot_indices(failed_robot_ids)
        x_si = uni_to_si_states(x_pose)
        positions = [world_to_grid(x_si[0, i], x_si[1, i]) for i in range(N_ROBOTS)]
        # for i in active_indices:
        #     robot = robots[i]
        #     robot.x, robot.y = positions[i]

        for i in range(N_ROBOTS):
            robots[i].x, robots[i].y = positions[i]
            robots[i].failed = i in failed_robot_ids

        active_positions = [positions[i] for i in active_indices]
        active_maps = [robots[i].local_covered for i in active_indices]
        sense_targets(active_positions, active_maps, covered, targets, found_targets)
        refresh_coverage_overlay(coverage_plot, covered)
        apply_decay(pher, TAU_DECAY, PHER_MIN)
        for gx, gy in active_positions:
            deposit_uniform(pher, gx, gy, PHER_DEPOSIT, robot_radius=ROBOT_RADIUS)
        for i in active_indices:
            robot = robots[i]
            if waypoint_timers[i] <= 0:
                start = positions[i]
                waypoint = robot.choose_move_search(
                    pher,
                    robots,
                    robot_radius=ROBOT_RADIUS,
                    collision_radius=COLLISION_RADIUS,
                )
                grid_goals[i] = apply_side_buffer(scale_grid_move(start, waypoint, grid_step_size))
                robot.x, robot.y = grid_goals[i]
                robot._last_move = (int(np.sign(waypoint[0] - start[0])), int(np.sign(waypoint[1] - start[1])))
                robot._position_visits[(robot.x, robot.y)] = robot._position_visits.get((robot.x, robot.y), 0) + 1
                waypoint_timers[i] = max(1, waypoint_update_steps)
            else:
                waypoint_timers[i] -= 1

        refresh_pheromone_overlay(pheromone_plot, pher)
        goals = make_goal_world(grid_goals)
        goals = hold_failed_robot_goals(goals, x_si, failed_robot_ids)
        prev_x_si = x_si.copy()
        x_pose = update_robotarium(r, x_pose, goals, helpers, failed_robot_ids=failed_robot_ids)
        moved_x_si = uni_to_si_states(x_pose)
        step += 1
        movement = np.linalg.norm(moved_x_si - prev_x_si, axis=0)
        for i in active_indices:
            if movement[i] < STUCK_MOVEMENT_EPS:
                stuck_counts[i] += 1
            else:
                stuck_counts[i] = 0
            if stuck_counts[i] >= STUCK_STEPS:
                waypoint_timers[i] = 0
                stuck_counts[i] = 0
        print_movement_report("Search", step, prev_x_si, moved_x_si, log_interval)
        refresh_found_plot(r, found_plot, found_targets, "Search Stigmergy Robotarium", step, len(targets))

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
        f"Search stigmergy Robotarium completed: found {len(found_targets)}/{len(targets)} "
        f"targets in {step} steps. Wall time is {wall_time() - start_wall_time} sec."
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
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--waypoint-update-steps", type=int, default=WAYPOINT_UPDATE_STEPS)
    parser.add_argument("--show-grid", type=lambda value: value.lower() == "true", default=True)
    args = parser.parse_args()
    run(
        show_figure=not args.no_show,
        max_steps=args.max_steps,
        grid_step_size=args.grid_step_size,
        log_interval=args.log_interval,
        waypoint_update_steps=args.waypoint_update_steps,
        show_grid=args.show_grid,
    )
