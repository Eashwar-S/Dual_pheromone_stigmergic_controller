import argparse
import random
import time
from typing import Dict, Iterator, Sequence, Tuple

import numpy as np

from robot_rendezvous import Robot, spiral_positions
from robotarium_swarm_common import *
import robotarium_swarm_common as common


def configure_experiment(search_robot_count: int) -> None:
    global N_SEARCH_ROBOTS, N_ROBOTS, SEARCH_IDS, TAU_DECAY
    common.configure_search_robot_count(search_robot_count)
    N_SEARCH_ROBOTS = common.N_SEARCH_ROBOTS
    N_ROBOTS = common.N_ROBOTS
    SEARCH_IDS = common.SEARCH_IDS
    TAU_DECAY = common.TAU_DECAY


def next_safe_spiral_goal(
    spiral: Iterator[Tuple[int, int]],
    search_positions: Sequence[Tuple[int, int]],
    fallback: Tuple[int, int],
    step_size: int = GRID_MOVE_CELLS,
) -> Tuple[int, int]:
    while True:
        candidate = None
        for _ in range(max(1, step_size)):
            candidate = next(spiral, None)
            if candidate is None:
                return fallback
        candidate = clamp_grid_goal(candidate)
        if all(
            max(
                abs(candidate[0] - search_position[0]),
                abs(candidate[1] - search_position[1]),
            ) > COLLISION_RADIUS
            for search_position in search_positions
        ):
            return candidate


def claim_target_parking_goal(
    position: Tuple[int, int],
    available_goals,
) -> Tuple[int, int]:
    goal = min(
        available_goals,
        key=lambda candidate: (
            abs(candidate[0] - position[0]) + abs(candidate[1] - position[1])
        ),
    )
    available_goals.remove(goal)
    return goal


def choose_search_goal(
    search_robot: Robot,
    robots,
    blue_pheromone: np.ndarray,
    pink_pheromone: np.ndarray,
) -> Tuple[int, int]:
    if (
        search_robot.mode == "FOLLOW"
        or search_robot.senses_attractive(
            pink_pheromone,
            sensing_radius=ATTRACTIVE_SENSING_RADIUS,
        )
    ):
        search_robot.mode = "FOLLOW"
        goal = search_robot.choose_move_follow(
            pink_pheromone,
            robots,
            collision_radius=COLLISION_RADIUS,
            sensing_radius=ATTRACTIVE_SENSING_RADIUS,
        )
        if goal != (search_robot.x, search_robot.y):
            return goal

    search_robot.mode = "SEARCH"
    return search_robot.choose_move_search(
        blue_pheromone,
        robots,
        robot_radius=ROBOT_RADIUS,
        collision_radius=COLLISION_RADIUS,
    )


def run(
    show_figure: bool = True,
    max_steps: int = MAX_STEPS,
    log_interval: int = 25,
    show_grid: bool = True,
    seed: int = START_RANDOM_SEED,
    search_distance: int = SEARCH_START_DISTANCE,
    search_robot_count: int = N_SEARCH_ROBOTS,
) -> Dict[str, object]:
    configure_experiment(search_robot_count)
    random.seed(ALGORITHM_RANDOM_SEED + seed)
    start_time = time.perf_counter()
    starts = initial_grid_positions(seed, search_distance)
    print(
        f"Rendezvous seed={seed}, advertiser_start={starts[ADVERTISER_ID]}, "
        f"search_distance={search_distance}, "
        f"search_starts={[starts[i] for i in SEARCH_IDS]}, target={TARGET_GRID}"
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
    x_si = uni_to_si(x_pose)

    robots = [
        Robot(
            id=i,
            x=starts[i][0],
            y=starts[i][1],
            local_covered=np.zeros((H, W), dtype=bool),
            start_x=starts[i][0],
            start_y=starts[i][1],
            mode="ADVERTISE" if i == ADVERTISER_ID else "SEARCH",
            advertising_target=TARGET_GRID if i == ADVERTISER_ID else None,
        )
        for i in range(N_ROBOTS)
    ]
    search_robots = [robots[i] for i in SEARCH_IDS]

    blue_pheromone = np.zeros((H, W), dtype=float)
    pink_pheromone = np.zeros((H, W), dtype=float)
    search_covered = np.zeros((H, W), dtype=bool)
    coverage_plot = add_coverage_overlay(r, search_covered, show_figure)
    overlays = add_pheromone_overlays(
        r,
        blue_pheromone,
        pink_pheromone,
        show_figure,
    )
    add_target_plot(r, show_figure)

    spiral = iter(spiral_positions(
        TARGET_GRID[0],
        TARGET_GRID[1],
        W,
        H,
        lane_spacing=SPIRAL_LANE_SPACING,
    ))
    next(spiral, TARGET_GRID)

    grid_goals = list(starts)
    grid_goals[ADVERTISER_ID] = next_safe_spiral_goal(
        spiral,
        [starts[i] for i in SEARCH_IDS],
        fallback=starts[ADVERTISER_ID],
    )
    available_parking_goals = [
        clamp_grid_goal((
            TARGET_GRID[0] + dx,
            TARGET_GRID[1] + dy,
        ))
        for dx, dy in TARGET_PARKING_OFFSETS[:len(SEARCH_IDS)]
    ]
    waypoint_timers = np.zeros(N_ROBOTS, dtype=int)
    stuck_counts = np.zeros(N_ROBOTS, dtype=int)
    found_robot_ids = set()
    step = 0

    while len(found_robot_ids) < len(SEARCH_IDS) and not should_stop(step, start_time, max_steps):
        x_si = uni_to_si(x_pose)
        positions = [
            world_to_grid(x_si[0, i], x_si[1, i])
            for i in range(N_ROBOTS)
        ]
        for i, robot in enumerate(robots):
            robot.x, robot.y = positions[i]

        advertiser = robots[ADVERTISER_ID]
        advertiser.deposit_attractive(
            pink_pheromone,
            target_x=TARGET_GRID[0],
            target_y=TARGET_GRID[1],
            attractive_radius=ATTRACTIVE_RADIUS,
        )

        apply_decay(blue_pheromone, TAU_DECAY)
        for robot_id, search_robot in zip(SEARCH_IDS, search_robots):
            if robot_id in found_robot_ids:
                continue
            mark_covered(
                search_covered,
                positions[robot_id][0],
                positions[robot_id][1],
                ROBOT_RADIUS,
            )
            search_robot.deposit_repulsive(
                blue_pheromone,
                PHER_DEPOSIT,
                ROBOT_RADIUS,
            )
            if target_found(positions[robot_id]):
                search_robot.mode = "FOUND"
                found_robot_ids.add(robot_id)
                grid_goals[robot_id] = claim_target_parking_goal(
                    positions[robot_id],
                    available_parking_goals,
                )
                waypoint_timers[robot_id] = 0
                stuck_counts[robot_id] = 0

        if has_arrived(x_si, ADVERTISER_ID, grid_goals[ADVERTISER_ID]) or stuck_counts[ADVERTISER_ID] >= STUCK_STEPS:
            grid_goals[ADVERTISER_ID] = next_safe_spiral_goal(
                spiral,
                [positions[i] for i in SEARCH_IDS],
                fallback=positions[ADVERTISER_ID],
            )
            stuck_counts[ADVERTISER_ID] = 0

        for robot_id, search_robot in zip(SEARCH_IDS, search_robots):
            if robot_id in found_robot_ids:
                continue
            if waypoint_timers[robot_id] <= 0 or stuck_counts[robot_id] >= STUCK_STEPS:
                old_position = positions[robot_id]
                chosen = choose_search_goal(
                    search_robot,
                    robots,
                    blue_pheromone,
                    pink_pheromone,
                )
                grid_goals[robot_id] = scale_grid_move(
                    old_position,
                    chosen,
                    GRID_MOVE_CELLS,
                )
                search_robot.x, search_robot.y = grid_goals[robot_id]
                search_robot._last_move = (
                    int(np.sign(chosen[0] - old_position[0])),
                    int(np.sign(chosen[1] - old_position[1])),
                )
                search_robot._position_visits[grid_goals[robot_id]] = (
                    search_robot._position_visits.get(grid_goals[robot_id], 0) + 1
                )
                waypoint_timers[robot_id] = max(1, WAYPOINT_UPDATE_STEPS)
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
        refresh_coverage_overlay(coverage_plot, search_covered)
        refresh_pheromone_overlays(
            overlays,
            blue_pheromone,
            pink_pheromone,
        )
        modes = [robot.mode for robot in search_robots]
        update_title(r, step, modes, show_figure)
        if log_interval > 0 and step % log_interval == 0:
            print_position_report(step, positions, x_pose, modes)

    final_si = uni_to_si(x_pose)
    final_positions = [
        world_to_grid(final_si[0, i], final_si[1, i])
        for i in range(N_ROBOTS)
    ]
    modes = [robot.mode for robot in search_robots]
    all_found = len(found_robot_ids) == len(SEARCH_IDS)
    print_position_report(step, final_positions, x_pose, modes)
    print(
        f"Rendezvous completed: all_found={all_found}, "
        f"found={len(found_robot_ids)}/{len(SEARCH_IDS)}, steps={step}, "
        f"wall_time={time.perf_counter() - start_time:.2f}s"
    )

    validation_errors = dict(getattr(r, "_errors", {}))
    finish_robotarium(r)
    return {
        "found": all_found,
        "found_robot_ids": sorted(found_robot_ids),
        "steps": step,
        "search_modes": modes,
        "starts": starts,
        "final_positions": final_positions,
        "validation_errors": validation_errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--show-grid", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=START_RANDOM_SEED)
    parser.add_argument(
        "--search-distance",
        type=int,
        default=SEARCH_START_DISTANCE,
        help="Exact Manhattan distance from the center target for search starts.",
    )
    parser.add_argument(
        "--search-robots",
        type=int,
        choices=range(1, 4),
        default=N_SEARCH_ROBOTS,
    )
    args = parser.parse_args()
    result = run(
        show_figure=not args.no_show,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        show_grid=args.show_grid,
        seed=args.seed,
        search_distance=args.search_distance,
        search_robot_count=args.search_robots,
    )
    if not result["found"]:
        raise SystemExit("Not all search robots reached the center target.")
