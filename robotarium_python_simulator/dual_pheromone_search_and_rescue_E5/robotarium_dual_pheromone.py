import argparse
import random
import time
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from robot_dual_pheromone import Robot, spiral_positions
from robotarium_swarm_common import *


def next_safe_spiral_goal(
    spiral: Iterator[Tuple[int, int]],
    other_positions: Sequence[Tuple[int, int]],
    fallback: Tuple[int, int],
    step_size: int = GRID_MOVE_CELLS,
) -> Tuple[int, int]:
    """Advance the advertiser spiral to a collision-safe waypoint."""
    while True:
        candidate = None
        for _ in range(max(1, step_size)):
            candidate = next(spiral, None)
            if candidate is None:
                return fallback
        candidate = clamp_grid_goal(candidate)
        if all(
            max(
                abs(candidate[0] - position[0]),
                abs(candidate[1] - position[1]),
            ) > COLLISION_RADIUS
            for position in other_positions
        ):
            return candidate


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
    """Run two searchers; the first finder advertises from a center spiral."""
    random.seed(ALGORITHM_RANDOM_SEED + seed)
    start_time = time.perf_counter()
    starts = initial_grid_positions(seed, search_distance)
    print(
        f"Dual-pheromone search seed={seed}, starts={starts}, "
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
            mode="SEARCH",
        )
        for robot_id in ROBOT_IDS
    ]

    repulsive_pheromone = np.zeros((H, W), dtype=float)
    attractive_pheromone = np.zeros((H, W), dtype=float)
    covered = np.zeros((H, W), dtype=bool)
    coverage_plot = add_coverage_overlay(r, covered, show_figure)
    pheromone_plots = add_pheromone_overlays(
        r,
        repulsive_pheromone,
        attractive_pheromone,
        show_figure,
    )
    target_plot = add_target_plot(r, show_figure)

    grid_goals = list(starts)
    waypoint_timers = np.zeros(N_ROBOTS, dtype=int)
    stuck_counts = np.zeros(N_ROBOTS, dtype=int)
    found_robot_ids = set()
    discovery_steps = {}
    advertiser_id: Optional[int] = None
    advertiser_spiral: Optional[Iterator[Tuple[int, int]]] = None
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
            mark_covered(
                robot.local_covered,
                robot.x,
                robot.y,
                ROBOT_RADIUS,
            )
            mark_covered(covered, robot.x, robot.y, ROBOT_RADIUS)

        newly_found = [
            robot_id
            for robot_id in ROBOT_IDS
            if (
                robot_id not in found_robot_ids
                and target_found(positions[robot_id])
            )
        ]
        for robot_id in newly_found:
            robot = robots[robot_id]
            robot.has_found_target = True
            robot.target_found_step = step
            found_robot_ids.add(robot_id)
            discovery_steps[robot_id] = step
            print(f"Robot {robot_id} found the target at step {step}.")

        if advertiser_id is None and newly_found:
            advertiser_id = min(newly_found)
            advertiser = robots[advertiser_id]
            advertiser.mode = "ADVERTISE"
            advertiser.advertising_target = TARGET_GRID
            advertiser_spiral = iter(spiral_positions(
                TARGET_GRID[0],
                TARGET_GRID[1],
                W,
                H,
                lane_spacing=SPIRAL_LANE_SPACING,
            ))
            next(advertiser_spiral, TARGET_GRID)
            other_positions = [
                positions[robot_id]
                for robot_id in ROBOT_IDS
                if robot_id != advertiser_id
            ]
            grid_goals[advertiser_id] = next_safe_spiral_goal(
                advertiser_spiral,
                other_positions,
                positions[advertiser_id],
                grid_step_size,
            )
            waypoint_timers[advertiser_id] = 0
            stuck_counts[advertiser_id] = 0
            print(f"Robot {advertiser_id} became the spiral advertiser.")

        if len(found_robot_ids) == N_ROBOTS:
            break

        apply_decay(repulsive_pheromone)
        for robot_id, robot in enumerate(robots):
            if robot_id == advertiser_id:
                robot.deposit_attractive(
                    attractive_pheromone,
                    target_x=TARGET_GRID[0],
                    target_y=TARGET_GRID[1],
                    attractive_radius=ATTRACTIVE_RADIUS,
                )
            else:
                robot.deposit_repulsive(
                    repulsive_pheromone,
                    PHER_DEPOSIT,
                    ROBOT_RADIUS,
                )

        if advertiser_id is not None and advertiser_spiral is not None:
            if (
                stuck_counts[advertiser_id] >= STUCK_STEPS
                or np.linalg.norm(
                    x_si[:, advertiser_id]
                    - np.array(grid_to_world(*grid_goals[advertiser_id]))
                )
                <= 0.65 * min(CELL_WIDTH, CELL_HEIGHT)
            ):
                other_positions = [
                    positions[robot_id]
                    for robot_id in ROBOT_IDS
                    if robot_id != advertiser_id
                ]
                grid_goals[advertiser_id] = next_safe_spiral_goal(
                    advertiser_spiral,
                    other_positions,
                    positions[advertiser_id],
                    grid_step_size,
                )
                stuck_counts[advertiser_id] = 0

        for robot_id, robot in enumerate(robots):
            if robot_id == advertiser_id or robot_id in found_robot_ids:
                continue
            if (
                waypoint_timers[robot_id] <= 0
                or stuck_counts[robot_id] >= STUCK_STEPS
            ):
                start = positions[robot_id]
                chosen = robot.choose_search_or_follow(
                    repulsive_pheromone,
                    attractive_pheromone,
                    robots,
                    robot_radius=ROBOT_RADIUS,
                    collision_radius=COLLISION_RADIUS,
                    attractive_sensing_radius=ATTRACTIVE_SENSING_RADIUS,
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
        modes = [robot.mode for robot in robots]
        refresh_coverage_overlay(coverage_plot, covered)
        refresh_pheromone_overlays(
            pheromone_plots,
            repulsive_pheromone,
            attractive_pheromone,
        )
        refresh_target_plot(target_plot, found_robot_ids)
        update_dual_title(
            r,
            step,
            modes,
            found_robot_ids,
            show_figure,
        )
        if log_interval > 0 and step % log_interval == 0:
            print_dual_position_report(
                step,
                positions,
                x_pose,
                modes,
                found_robot_ids,
                repulsive_pheromone,
                attractive_pheromone,
            )

    final_si = uni_to_si(x_pose)
    final_positions = [
        world_to_grid(final_si[0, robot_id], final_si[1, robot_id])
        for robot_id in ROBOT_IDS
    ]
    modes = [robot.mode for robot in robots]
    all_found = len(found_robot_ids) == N_ROBOTS
    wall_seconds = time.perf_counter() - start_time
    refresh_target_plot(target_plot, found_robot_ids)
    update_dual_title(r, step, modes, found_robot_ids, show_figure)
    print_dual_position_report(
        step,
        final_positions,
        x_pose,
        modes,
        found_robot_ids,
        repulsive_pheromone,
        attractive_pheromone,
    )
    print(
        f"Dual-pheromone search completed: all_found={all_found}, "
        f"found={len(found_robot_ids)}/{N_ROBOTS}, "
        f"advertiser={advertiser_id}, steps={step}, "
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
        "advertiser_id": advertiser_id,
        "steps": step,
        "modes": modes,
        "starts": starts,
        "final_positions": final_positions,
        "repulsive_max": float(np.max(repulsive_pheromone)),
        "attractive_max": float(np.max(attractive_pheromone)),
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
        print(
            "The run reached its stopping condition before both robots "
            "found the center target; Robotarium cleanup completed normally."
        )
