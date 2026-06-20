import argparse
import random
import time
from typing import Dict

import numpy as np

from robot_single_pheromone import Robot
from robotarium_swarm_common import *


def run(
    show_figure: bool = True,
    max_steps: int = MAX_STEPS,
    log_interval: int = 25,
    show_grid: bool = True,
    seed: int = START_RANDOM_SEED,
    search_distance: int = SEARCH_START_DISTANCE,
    grid_step_size: int = GRID_MOVE_CELLS,
    waypoint_update_steps: int = WAYPOINT_UPDATE_STEPS,
) -> Dict[str, object]:
    """Run two independent searchers using one repulsive pheromone field."""
    show_figure = resolve_show_figure(show_figure)
    random.seed(ALGORITHM_RANDOM_SEED + seed)
    start_time = time.perf_counter()
    starts = initial_grid_positions(seed, search_distance)
    print(
        f"Single-pheromone search seed={seed}, starts={starts}, "
        f"target={TARGET_GRID}, search_distance={search_distance}"
    )

    r = create_robotarium(
        show_figure=show_figure,
        show_grid=show_grid,
        seed=seed,
        search_distance=search_distance,
    )
    helpers = create_motion_helpers()
    _, _, _, uni_to_si = helpers
    x_pose = r.get_poses()

    robots = [
        Robot(
            id=robot_id,
            x=starts[robot_id][0],
            y=starts[robot_id][1],
            local_covered=np.zeros((H, W), dtype=bool),
        )
        for robot_id in ROBOT_IDS
    ]

    pheromone = np.zeros((H, W), dtype=float)
    covered = np.zeros((H, W), dtype=bool)
    coverage_plot = add_coverage_overlay(r, covered, show_figure)
    pheromone_plot = add_pheromone_overlay(r, pheromone, show_figure)
    target_plot = add_target_plot(r, show_figure)

    grid_goals = list(starts)
    waypoint_timers = np.zeros(N_ROBOTS, dtype=int)
    stuck_counts = np.zeros(N_ROBOTS, dtype=int)
    found_robot_ids = set()
    discovery_steps = {}
    step = 0

    while (
        len(found_robot_ids) < N_ROBOTS
        and not should_stop(step, start_time, max_steps)
    ):
        x_si = uni_to_si(x_pose)
        positions = [
            world_to_grid(x_si[0, robot_id], x_si[1, robot_id])
            for robot_id in ROBOT_IDS
        ]
        for robot_id, robot in enumerate(robots):
            robot.x, robot.y = positions[robot_id]
            robot.mode = "SEARCH"
            mark_covered(
                robot.local_covered,
                robot.x,
                robot.y,
                ROBOT_RADIUS,
            )
            mark_covered(covered, robot.x, robot.y, ROBOT_RADIUS)

            if (
                robot_id not in found_robot_ids
                and target_found(positions[robot_id])
            ):
                robot.has_found_target = True
                robot.target_found_step = step
                found_robot_ids.add(robot_id)
                discovery_steps[robot_id] = step
                print(f"Robot {robot_id} found the target at step {step}.")

        apply_decay(pheromone)
        for robot in robots:
            robot.deposit_pheromone(
                pheromone,
                PHER_DEPOSIT,
                ROBOT_RADIUS,
            )

        for robot_id, robot in enumerate(robots):
            if (
                (
                    waypoint_timers[robot_id] <= 0
                    and has_arrived(x_si, robot_id, grid_goals[robot_id])
                )
                or stuck_counts[robot_id] >= STUCK_STEPS
            ):
                start = positions[robot_id]
                chosen = robot.choose_move_search(
                    pheromone,
                    robots,
                    robot_radius=ROBOT_RADIUS,
                    collision_radius=COLLISION_RADIUS,
                    step_size=grid_step_size,
                )
                waypoint = scale_grid_move(
                    start,
                    chosen,
                    grid_step_size,
                )
                grid_goals[robot_id] = waypoint
                robot.record_waypoint(start, chosen, waypoint)
                waypoint_timers[robot_id] = max(
                    1,
                    waypoint_update_steps,
                )
                stuck_counts[robot_id] = 0
            else:
                waypoint_timers[robot_id] -= 1

        previous_si = x_si.copy()
        x_pose = update_robotarium(
            r,
            x_pose,
            grid_goals,
            helpers,
        )
        moved_si = uni_to_si(x_pose)
        movement = np.linalg.norm(moved_si - previous_si, axis=0)
        stuck_counts = np.where(
            movement < STUCK_MOVEMENT_EPS,
            stuck_counts + 1,
            0,
        )

        step += 1
        refresh_coverage_overlay(coverage_plot, covered)
        refresh_pheromone_overlay(pheromone_plot, pheromone)
        refresh_target_plot(target_plot, found_robot_ids)
        update_title(r, step, found_robot_ids, show_figure)
        if log_interval > 0 and step % log_interval == 0:
            print_position_report(
                step,
                positions,
                x_pose,
                found_robot_ids,
                pheromone,
            )

    final_si = uni_to_si(x_pose)
    final_positions = [
        world_to_grid(final_si[0, robot_id], final_si[1, robot_id])
        for robot_id in ROBOT_IDS
    ]
    all_found = len(found_robot_ids) == N_ROBOTS
    wall_seconds = time.perf_counter() - start_time
    print_position_report(
        step,
        final_positions,
        x_pose,
        found_robot_ids,
        pheromone,
    )
    print(
        f"Single-pheromone search completed: all_found={all_found}, "
        f"found={len(found_robot_ids)}/{N_ROBOTS}, steps={step}, "
        f"wall_time={wall_seconds:.2f}s"
    )

    if all_found:
        add_completion_banner(r, step, wall_seconds, show_figure)

    validation_errors = dict(getattr(r, "_errors", {}))
    finish_robotarium(r)
    return {
        "found": all_found,
        "found_robot_ids": sorted(found_robot_ids),
        "discovery_steps": discovery_steps,
        "steps": step,
        "search_modes": [robot.mode for robot in robots],
        "starts": starts,
        "final_positions": final_positions,
        "pheromone_min": float(np.min(pheromone)),
        "pheromone_max": float(np.max(pheromone)),
        "validation_errors": validation_errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--show-grid",
        type=lambda value: value.lower() == "true",
        default=True,
    )
    parser.add_argument("--seed", type=int, default=START_RANDOM_SEED)
    parser.add_argument(
        "--search-distance",
        type=int,
        default=SEARCH_START_DISTANCE,
        help="Exact Manhattan distance of both starts from the center.",
    )
    parser.add_argument(
        "--grid-step-size",
        type=int,
        default=GRID_MOVE_CELLS,
    )
    parser.add_argument(
        "--waypoint-update-steps",
        type=int,
        default=WAYPOINT_UPDATE_STEPS,
    )
    args = parser.parse_args()
    result = run(
        show_figure=not args.no_show,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        show_grid=args.show_grid,
        seed=args.seed,
        search_distance=args.search_distance,
        grid_step_size=args.grid_step_size,
        waypoint_update_steps=args.waypoint_update_steps,
    )
    if not result["found"]:
        raise SystemExit("Both robots did not reach the center target.")
