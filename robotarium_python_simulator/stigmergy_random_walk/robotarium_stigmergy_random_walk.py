import argparse
from pathlib import Path
from typing import Dict, Set, Tuple
import sys

import numpy as np
from robot_random import Robot
from robotarium_swarm_common import *

ROBOTARIUM_RANDOM_WALK_GRID_STEP_CELLS = ROBOT_RADIUS + 2
ROBOTARIUM_RANDOM_WALK_ARRIVAL_CELLS = 1.25


def clamp_robotarium_goal(goal: Tuple[int, int]) -> Tuple[int, int]:
    margin = ROBOT_RADIUS
    gx, gy = goal
    return (
        int(np.clip(gx, margin, W - 1 - margin)),
        int(np.clip(gy, margin, H - 1 - margin)),
    )


def create_random_walk_motion_helpers():
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


def run(show_figure: bool = True, max_steps: int = MAX_STEPS,
        grid_step_size: int = ROBOTARIUM_RANDOM_WALK_GRID_STEP_CELLS,
        log_interval: int = 25,
        waypoint_update_steps: int = WAYPOINT_UPDATE_STEPS,
        show_grid: bool = True) -> Dict[str, object]:
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
    _, _, _, uni_to_si_states = helpers
    x_si = uni_to_si_states(x_pose)

    robots = [
        Robot(
            id=i,
            x=world_to_grid(x_si[0, i], x_si[1, i])[0],
            y=world_to_grid(x_si[0, i], x_si[1, i])[1],
            robot_radius=ROBOT_RADIUS,
            collision_radius=COLLISION_RADIUS,
            local_covered=np.zeros((H, W), dtype=bool),
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
    pher = np.zeros((H, W), dtype=float)
    grid_goals = [clamp_robotarium_goal((robot.x, robot.y)) for robot in robots]
    pheromone_plot = add_pheromone_overlay(r, pher, show_figure)
    found_plot = add_target_plot(r, targets, found_targets, "Random-Walk Stigmergy Robotarium", 0, show_figure)

    step = 0
    while step < max_steps and len(found_targets) < len(targets):
        update_failed_robots(step, failure_schedule, failed_robot_ids, "Random-walk")
        active_indices = active_robot_indices(failed_robot_ids)
        x_si = uni_to_si_states(x_pose)
        positions = [world_to_grid(x_si[0, i], x_si[1, i]) for i in range(N_ROBOTS)]
        for i in active_indices:
            robot = robots[i]
            robot.x, robot.y = positions[i]

        active_positions = [positions[i] for i in active_indices]
        active_maps = [robots[i].local_covered for i in active_indices]
        sense_targets(active_positions, active_maps, covered, targets, found_targets)
        apply_decay(pher, TAU_DECAY, PHER_MIN)
        for gx, gy in active_positions:
            deposit_uniform(pher, gx, gy, PHER_DEPOSIT, robot_radius=ROBOT_RADIUS)
        for i in active_indices:
            robot = robots[i]
            if has_arrived(x_si, i, grid_goals[i], tolerance_cells=ROBOTARIUM_RANDOM_WALK_ARRIVAL_CELLS):
                start = positions[i]
                waypoint = robot.choose_move(pher, robot_rng, None)
                grid_goals[i] = clamp_robotarium_goal(scale_grid_move(start, waypoint, grid_step_size))
                robot.x, robot.y = grid_goals[i]

        refresh_pheromone_overlay(pheromone_plot, pher)
        goals = make_goal_world(grid_goals)
        goals = hold_failed_robot_goals(goals, x_si, failed_robot_ids)
        prev_x_si = x_si.copy()
        x_pose = update_robotarium(r, x_pose, goals, helpers, failed_robot_ids=failed_robot_ids)
        moved_x_si = uni_to_si_states(x_pose)
        step += 1
        print_movement_report("Random-walk", step, prev_x_si, moved_x_si, log_interval)
        refresh_found_plot(r, found_plot, found_targets, "Random-Walk Stigmergy Robotarium", step, len(targets))

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
        f"Random-walk stigmergy Robotarium completed: found {len(found_targets)}/{len(targets)} "
        f"targets in {step} steps. Max displacement: {displacement.max():.3f} m."
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
    args = parser.parse_args()
    run(
        show_figure=not args.no_show,
        max_steps=args.max_steps,
        grid_step_size=args.grid_step_size,
        log_interval=args.log_interval,
        waypoint_update_steps=args.waypoint_update_steps,
        show_grid=args.show_grid,
    )
