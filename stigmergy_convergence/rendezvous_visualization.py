from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from common.simulation import FrameWriter, run_animation
from common.utilities import discover_targets_in_vnhood, mark_visible
from common.visualization import coverage_to_image, pheromone_to_rgba
import config_experiments
from robot_rendezvous import Robot, spiral_positions
from stigmergy_common.pheromone import apply_decay


MIN_START_DISTANCE_TO_TARGET = 20


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


def generate_search_starts(
    grid_size: int,
    n_search_robots: int,
    target: Tuple[int, int],
    requested_distance_shell: int,
    min_distance: int = MIN_START_DISTANCE_TO_TARGET,
) -> List[Tuple[int, int]]:
    all_cells = [
        (x, y)
        for x in range(grid_size)
        for y in range(grid_size)
        if (x, y) != target and manhattan((x, y), target) >= min_distance
    ]
    exact_shell = [p for p in all_cells if manhattan(p, target) == requested_distance_shell]
    if len(exact_shell) >= n_search_robots:
        return _evenly_spaced_cells(exact_shell, target, n_search_robots)

    distance_candidates = sorted(
        {manhattan(p, target) for p in all_cells},
        key=lambda distance: (abs(distance - requested_distance_shell), -distance),
    )
    for candidate_distance in distance_candidates:
        candidates = [p for p in all_cells if manhattan(p, target) == candidate_distance]
        if len(candidates) >= n_search_robots:
            return _evenly_spaced_cells(candidates, target, n_search_robots)
    return _evenly_spaced_cells(all_cells, target, n_search_robots)


def make_robots(grid_size: int, search_starts: List[Tuple[int, int]], target: Tuple[int, int]) -> List[Robot]:
    H = W = grid_size
    advertiser = Robot(
        id=0,
        x=target[0],
        y=target[1],
        start_x=target[0],
        start_y=target[1],
        local_covered=np.zeros((H, W), dtype=bool),
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
            local_covered=np.zeros((H, W), dtype=bool),
            mode="SEARCH",
        )
        for i, start in enumerate(search_starts)
    ]
    return [advertiser, *searchers]


def make_convergence_metrics(searchers: List[Robot], target: Tuple[int, int]) -> Dict[int, Dict[str, object]]:
    metrics: Dict[int, Dict[str, object]] = {}
    for searcher in searchers:
        initial_distance = manhattan((searcher.x, searcher.y), target)
        metrics[searcher.id] = {
            "initial_distance": initial_distance,
            "time_to_attractive_detection": None,
            "distance_at_detection": np.nan,
            "mode_switch_step": None,
            "found_step": None,
            "previous_distance": initial_distance,
            "final_distance": initial_distance,
            "stagnation_count": 0,
            "post_detection_stagnation_count": 0,
            "distance_history": [initial_distance],
        }
    return metrics


def sim_step(
    *,
    current_step: int,
    robots,
    spiral_iter,
    pher_rep: np.ndarray,
    pher_attr: np.ndarray,
    covered: np.ndarray,
    targets: Set[Tuple[int, int]],
    found_targets: Set[Tuple[int, int]],
    global_target_visits: Dict[Tuple[int, int], Set[int]],
    convergence_metrics: Dict[int, Dict[str, object]],
    robot_radius: int,
    collision_radius: int,
    pher_deposit: float,
    tau_decay: float,
    pher_min: float,
    robot_attractive_radius: int,
) -> Tuple[bool, List[Robot]]:
    advertiser = robots[0]
    searchers = robots[1:]
    H, W = pher_rep.shape
    target = next(iter(targets))

    apply_decay(pher_rep, tau_decay, pher_min)

    advertiser.deposit_attractive(pher_attr, target[0], target[1], robot_attractive_radius)
    try:
        advertiser.x, advertiser.y = next(spiral_iter)
    except StopIteration:
        pass

    for searcher in searchers:
        metrics = convergence_metrics[searcher.id]
        if searcher.mode == "FOUND":
            distance = manhattan((searcher.x, searcher.y), target)
            metrics["distance_history"].append(distance)
            metrics["final_distance"] = distance
            continue

        previous_mode = searcher.mode
        current_distance = manhattan((searcher.x, searcher.y), target)
        if searcher.senses_attractive(pher_attr, robot_attractive_radius) and metrics["time_to_attractive_detection"] is None:
            metrics["time_to_attractive_detection"] = current_step
            metrics["distance_at_detection"] = current_distance

        mark_visible(searcher.local_covered, searcher.x, searcher.y, robot_radius)
        mark_visible(covered, searcher.x, searcher.y, robot_radius)
        discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, robot_radius)

        if target in found_targets:
            searcher.mode = "FOUND"
        else:
            searcher.step_search_or_follow(
                pher_rep=pher_rep,
                pher_attr=pher_attr,
                targets=targets,
                all_robots=robots,
                global_target_visits=global_target_visits,
                robot_radius=robot_radius,
                collision_radius=collision_radius,
                pher_deposit=pher_deposit,
                attractive_sensing_radius=robot_attractive_radius,
            )

        if previous_mode == "SEARCH" and searcher.mode == "FOLLOW" and metrics["mode_switch_step"] is None:
            metrics["mode_switch_step"] = current_step

        mark_visible(searcher.local_covered, searcher.x, searcher.y, robot_radius)
        mark_visible(covered, searcher.x, searcher.y, robot_radius)
        discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, robot_radius)

        new_distance = manhattan((searcher.x, searcher.y), target)
        if new_distance >= metrics["previous_distance"]:
            metrics["stagnation_count"] += 1
            if metrics["time_to_attractive_detection"] is not None:
                metrics["post_detection_stagnation_count"] += 1
        metrics["previous_distance"] = new_distance
        metrics["final_distance"] = new_distance
        metrics["distance_history"].append(new_distance)

        if searcher.mode == "FOUND" or target in found_targets:
            searcher.mode = "FOUND"
            if metrics["found_step"] is None:
                metrics["found_step"] = current_step

    found_searchers = [r for r in searchers if r.mode == "FOUND"]
    return bool(found_searchers), found_searchers


def print_convergence_metrics(
    *,
    searchers: List[Robot],
    convergence_metrics: Dict[int, Dict[str, object]],
    target: Tuple[int, int],
    search_starts: List[Tuple[int, int]],
    requested_distance_shell: int,
    final_step: int,
) -> None:
    optimal_manhattan_steps = min(manhattan(start, target) for start in search_starts)
    print(
        f"steps={final_step}, requested_distance_shell={requested_distance_shell}, "
        f"optimal_manhattan_steps={optimal_manhattan_steps}, search_starts={search_starts}, target={target}"
    )
    for searcher in searchers:
        metrics = convergence_metrics[searcher.id]
        found_step = metrics["found_step"]
        detection_step = metrics["time_to_attractive_detection"]
        time_from_detection = found_step - detection_step if found_step is not None and detection_step is not None else np.nan
        path_efficiency = (
            metrics["initial_distance"] / found_step if found_step is not None and found_step > 0 else np.nan
        )
        post_detection_path_efficiency = (
            metrics["distance_at_detection"] / time_from_detection
            if time_from_detection and time_from_detection > 0
            else np.nan
        )
        convergence_rate = (
            (metrics["distance_at_detection"] - metrics["final_distance"]) / time_from_detection
            if detection_step is not None and time_from_detection and time_from_detection > 0
            else np.nan
        )
        print(
            f"robot={searcher.id}, found_step={found_step}, "
            f"time_to_attractive_detection={detection_step}, "
            f"time_from_detection_to_target={time_from_detection}, "
            f"mode_switch_step={metrics['mode_switch_step']}, "
            f"path_efficiency={path_efficiency}, "
            f"post_detection_path_efficiency={post_detection_path_efficiency}, "
            f"convergence_rate={convergence_rate}, "
            f"stagnation_count={metrics['stagnation_count']}, "
            f"post_detection_stagnation_count={metrics['post_detection_stagnation_count']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-robots", type=int, default=1, help="Number of search robots to visualize.")
    parser.add_argument("--distance-shell", type=int, default=20, help="Requested Manhattan shell for search robot starts.")
    parser.add_argument("--grid-size", type=int, default=100, help="Grid side length.")
    args = parser.parse_args()

    GRID_SIZE = max(25, args.grid_size)
    N_SEARCH_ROBOTS = max(1, args.search_robots)
    REQUESTED_DISTANCE_SHELL = max(MIN_START_DISTANCE_TO_TARGET, args.distance_shell)
    RANDOM_SEED = config_experiments.BASE_SEED
    ROBOT_RADIUS = config_experiments.ROBOT_RADIUS
    COLLISION_RADIUS = 1
    SPIRAL_LANE_SPACING = max(1, 2 * ROBOT_RADIUS)
    PHER_DEPOSIT = 1.0
    TAU_DECAY = (GRID_SIZE**2) / 2
    PHER_MIN = 1e-6
    STEPS_PER_FRAME = 30
    INTERVAL_MS = 50
    MAX_STEPS = 20000
    OUTPUT_DIR = Path("output_frames/rendezvous")

    random.seed(RANDOM_SEED)

    W = H = GRID_SIZE
    target = (W // 2, H // 2)
    search_starts = generate_search_starts(GRID_SIZE, N_SEARCH_ROBOTS, target, REQUESTED_DISTANCE_SHELL)
    optimal_manhattan_steps = min(manhattan(start, target) for start in search_starts)

    robots = make_robots(GRID_SIZE, search_starts, target)
    advertiser = robots[0]
    searchers = robots[1:]
    convergence_metrics = make_convergence_metrics(searchers, target)
    spiral_iter = spiral_positions(target[0], target[1], W, H, lane_spacing=SPIRAL_LANE_SPACING)
    next(spiral_iter)

    pher_rep = np.zeros((H, W), dtype=float)
    pher_attr = np.zeros((H, W), dtype=float)
    covered = np.zeros((H, W), dtype=bool)
    targets = {target}
    found_targets: Set[Tuple[int, int]] = set()
    global_target_visits: Dict[Tuple[int, int], Set[int]] = {}

    frame_writer = FrameWriter(str(OUTPUT_DIR))
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.18)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_rep = fig.add_subplot(gs[0, 1])
    ax_attr = fig.add_subplot(gs[0, 2])

    world_img = ax_world.imshow(coverage_to_image(covered), origin="lower", extent=[0, W, 0, H], vmin=0.0, vmax=1.0)
    rep_img = ax_rep.imshow(pheromone_to_rgba(pher_rep), origin="lower", extent=[0, W, 0, H])
    attr_img = ax_attr.imshow(pheromone_to_rgba(pher_attr), origin="lower", extent=[0, W, 0, H])

    tick_step = 10 if W <= 100 else 20
    for ax in [ax_world, ax_rep, ax_attr]:
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, W + 1, tick_step))
        ax.set_yticks(np.arange(0, H + 1, tick_step))
        ax.grid(which="major", color="k", alpha=0.15, linewidth=0.5)

    ax_world.set_title("World - Rendezvous Coverage")
    ax_rep.set_title("Repulsive Pheromone")
    ax_attr.set_title("Attractive Spiral Pheromone")

    target_plot = ax_world.scatter([target[0] + 0.5], [target[1] + 0.5], s=70, marker="x", c="red", zorder=5)
    robot_scat = ax_world.scatter(
        [r.x + 0.5 for r in robots],
        [r.y + 0.5 for r in robots],
        s=[45 for _ in robots],
        c=["blue", *(["black"] * len(searchers))],
        zorder=6,
    )
    labels = [ax_world.text(advertiser.x + 0.6, advertiser.y + 0.6, "A", fontsize=8, color="blue", zorder=7)]
    labels.extend(
        ax_world.text(r.x + 0.6, r.y + 0.6, f"S{r.id}", fontsize=8, color="black", zorder=7)
        for r in searchers
    )

    state = {"step": 0, "done": False, "printed": False}

    def update(frame):
        if state["done"]:
            return (world_img, rep_img, attr_img, robot_scat, target_plot, *labels)

        for _ in range(STEPS_PER_FRAME):
            state["step"] += 1
            done, found_searchers = sim_step(
                current_step=state["step"],
                robots=robots,
                spiral_iter=spiral_iter,
                pher_rep=pher_rep,
                pher_attr=pher_attr,
                covered=covered,
                targets=targets,
                found_targets=found_targets,
                global_target_visits=global_target_visits,
                convergence_metrics=convergence_metrics,
                robot_radius=ROBOT_RADIUS,
                collision_radius=COLLISION_RADIUS,
                pher_deposit=PHER_DEPOSIT,
                tau_decay=TAU_DECAY,
                pher_min=PHER_MIN,
                robot_attractive_radius=ROBOT_RADIUS,
            )
            if done or state["step"] >= MAX_STEPS:
                state["done"] = True
                if not state["printed"]:
                    print_convergence_metrics(
                        searchers=searchers,
                        convergence_metrics=convergence_metrics,
                        target=target,
                        search_starts=search_starts,
                        requested_distance_shell=REQUESTED_DISTANCE_SHELL,
                        final_step=state["step"],
                    )
                    state["printed"] = True
                break

        world_img.set_data(coverage_to_image(covered))
        rep_img.set_data(pheromone_to_rgba(pher_rep))
        attr_img.set_data(pheromone_to_rgba(pher_attr))
        robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
        robot_scat.set_facecolors(["blue", *(["green" if r.mode == "FOUND" else "black" for r in searchers])])
        for label, r in zip(labels, robots):
            label.set_position((r.x + 0.6, r.y + 0.6))

        distances = [manhattan((searcher.x, searcher.y), target) for searcher in searchers]
        ax_world.set_title(
            f"World - step={state['step']}, searchers={N_SEARCH_ROBOTS}, "
            f"min_d={min(distances)}, mean_d={float(np.mean(distances)):.1f}, optimal={optimal_manhattan_steps}"
        )
        frame_writer.save(fig)
        return (world_img, rep_img, attr_img, robot_scat, target_plot, *labels)

    anim = run_animation(fig, update, frames=MAX_STEPS, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
