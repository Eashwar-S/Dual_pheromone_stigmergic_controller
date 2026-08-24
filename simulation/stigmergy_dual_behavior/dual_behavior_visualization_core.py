from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set, Tuple
import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.simulation import FrameWriter, run_animation
from common.utilities import discover_targets_in_vnhood, mark_visible
from common.visualization import coverage_to_image, pheromone_to_rgba
from stigmergy_common.pheromone import apply_decay

from dual_behavior_core import (
    COLLISION_RADIUS,
    MIN_START_DISTANCE_TO_TARGET,
    PHER_DEPOSIT,
    PHER_MIN,
    ROBOT_RADIUS,
    SPIRAL_LANE_SPACING,
    VARIANTS,
    SearchRescueRobot,
    generate_spawn_position_sets,
    make_robots,
    manhattan,
    spiral_positions,
    _deposit_advertisers,
)
import config_experiments


def sim_step(
    *,
    variant_name: str,
    current_step: int,
    robots: list[SearchRescueRobot],
    advertiser_spirals: Dict[int, Iterable[Tuple[int, int]]],
    pher_search: np.ndarray,
    pher_follow: np.ndarray,
    covered: np.ndarray,
    targets: Set[Tuple[int, int]],
    found_targets: Set[Tuple[int, int]],
    global_target_visits: Dict[Tuple[int, int], Set[int]],
    convergence_metrics: Dict[int, Dict[str, object]],
    tau_decay: float,
) -> bool:
    variant = VARIANTS[variant_name]
    H, W = pher_search.shape
    target = next(iter(targets))

    if variant.evaporate_search_field:
        apply_decay(pher_search, tau_decay, PHER_MIN)

    _deposit_advertisers(
        variant=variant,
        robots=robots,
        advertiser_spirals=advertiser_spirals,
        pher_follow=pher_follow,
        target=target,
        grid_size=W,
    )
    attraction_enabled = variant.rescue_mode and any(r.mode == "ADVERTISE" for r in robots)

    for searcher in robots:
        metrics = convergence_metrics[searcher.id]
        if searcher.mode in {"FOUND", "ADVERTISE"}:
            distance = manhattan((searcher.x, searcher.y), target)
            metrics["distance_history"].append(distance)
            metrics["final_distance"] = distance
            continue

        previous_mode = searcher.mode
        current_distance = manhattan((searcher.x, searcher.y), target)
        if (
            variant.rescue_mode
            and variant.has_attractive_channel
            and attraction_enabled
            and searcher.senses_attractive(pher_follow, ROBOT_RADIUS)
            and metrics["time_to_attractive_detection"] is None
        ):
            metrics["time_to_attractive_detection"] = current_step
            metrics["distance_at_detection"] = current_distance

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

        if previous_mode == "SEARCH" and searcher.mode == "FOLLOW" and metrics["mode_switch_step"] is None:
            metrics["mode_switch_step"] = current_step

        mark_visible(searcher.local_covered, searcher.x, searcher.y, ROBOT_RADIUS)
        mark_visible(covered, searcher.x, searcher.y, ROBOT_RADIUS)
        target_visible = manhattan((searcher.x, searcher.y), target) <= ROBOT_RADIUS
        discover_targets_in_vnhood(searcher.x, searcher.y, targets, found_targets, W, H, ROBOT_RADIUS)
        if target_visible and target not in searcher.visited_targets:
            can_advertise = variant.rescue_mode and not any(r.mode == "ADVERTISE" for r in robots)
            searcher.mark_target_found(target, global_target_visits, can_advertise=can_advertise)

        new_distance = manhattan((searcher.x, searcher.y), target)
        if new_distance >= metrics["previous_distance"]:
            metrics["stagnation_count"] += 1
            if metrics["time_to_attractive_detection"] is not None:
                metrics["post_detection_stagnation_count"] += 1
        metrics["previous_distance"] = new_distance
        metrics["final_distance"] = new_distance
        metrics["distance_history"].append(new_distance)

        if target in searcher.visited_targets and metrics["found_step"] is None:
            metrics["found_step"] = current_step

    return all(target in r.visited_targets for r in robots)


def make_convergence_metrics(searchers: list[SearchRescueRobot], target: Tuple[int, int]) -> Dict[int, Dict[str, object]]:
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


def print_metrics(
    *,
    variant_name: str,
    searchers: list[SearchRescueRobot],
    convergence_metrics: Dict[int, Dict[str, object]],
    target: Tuple[int, int],
    search_starts: list[Tuple[int, int]],
    requested_distance_shell: int,
    final_step: int,
) -> None:
    optimal_manhattan_steps = min(manhattan(start, target) for start in search_starts)
    all_found = all(target in searcher.visited_targets for searcher in searchers)
    print(
        f"variant={variant_name}, all_found={all_found}, steps={final_step}, "
        f"requested_distance_shell={requested_distance_shell}, "
        f"optimal_manhattan_steps={optimal_manhattan_steps}, starts={search_starts}, target={target}"
    )
    for searcher in searchers:
        metrics = convergence_metrics[searcher.id]
        print(
            f"robot={searcher.id}, mode={searcher.mode}, found_step={metrics['found_step']}, "
            f"time_to_attractive_detection={metrics['time_to_attractive_detection']}, "
            f"mode_switch_step={metrics['mode_switch_step']}, "
            f"current_distance={metrics['final_distance']}, "
            f"target_arrival_distance={0 if metrics['found_step'] is not None else metrics['final_distance']}, "
            f"stagnation_count={metrics['stagnation_count']}, "
            f"post_detection_stagnation_count={metrics['post_detection_stagnation_count']}"
        )


def main_visualization(variant_name: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-robots", type=int, default=2, help="Number of search robots to visualize.")
    parser.add_argument("--distance-shell", type=int, default=20, help="Requested Manhattan shell for starts.")
    parser.add_argument("--grid-size", type=int, default=100, help="Grid side length.")
    args = parser.parse_args()

    variant = VARIANTS[variant_name]
    grid_size = max(25, args.grid_size)
    n_search_robots = max(1, args.search_robots)
    requested_distance_shell = max(MIN_START_DISTANCE_TO_TARGET, args.distance_shell)
    random.seed(config_experiments.BASE_SEED)

    W = H = grid_size
    target = (W // 2, H // 2)
    spawn_sets = generate_spawn_position_sets(grid_size, (requested_distance_shell,), n_search_robots)
    _, start_seed, search_starts = spawn_sets[0]
    robots = make_robots(grid_size, search_starts)
    convergence_metrics = make_convergence_metrics(robots, target)
    advertiser_spirals: Dict[int, Iterable[Tuple[int, int]]] = {}

    pher_search = np.zeros((H, W), dtype=float)
    pher_follow = pher_search if variant.merged_field else np.zeros((H, W), dtype=float)
    covered = np.zeros((H, W), dtype=bool)
    targets = {target}
    found_targets: Set[Tuple[int, int]] = set()
    global_target_visits: Dict[Tuple[int, int], Set[int]] = {}
    tau_decay = (grid_size**2) / 2
    print(f"start_seed={start_seed}, search_starts={search_starts}")

    steps_per_frame = 30
    interval_ms = 50
    max_steps = 20000
    output_dir = Path("output_frames") / variant.output_name

    frame_writer = FrameWriter(str(output_dir))
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.18)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_search = fig.add_subplot(gs[0, 1])
    ax_follow = fig.add_subplot(gs[0, 2])

    world_img = ax_world.imshow(coverage_to_image(covered), origin="lower", extent=[0, W, 0, H], vmin=0.0, vmax=1.0)
    search_img = ax_search.imshow(pheromone_to_rgba(pher_search), origin="lower", extent=[0, W, 0, H])
    follow_img = ax_follow.imshow(pheromone_to_rgba(pher_follow), origin="lower", extent=[0, W, 0, H])

    tick_step = 10 if W <= 100 else 20
    for ax in [ax_world, ax_search, ax_follow]:
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, W + 1, tick_step))
        ax.set_yticks(np.arange(0, H + 1, tick_step))
        ax.grid(which="major", color="k", alpha=0.15, linewidth=0.5)

    fig.suptitle(variant.name, fontsize=12)
    ax_world.set_title("World", fontsize=10)
    ax_search.set_title("Search field", fontsize=10)
    ax_follow.set_title("Follow field", fontsize=10)

    target_plot = ax_world.scatter([target[0] + 0.5], [target[1] + 0.5], s=70, marker="x", c="red", zorder=5)
    robot_scat = ax_world.scatter(
        [r.x + 0.5 for r in robots],
        [r.y + 0.5 for r in robots],
        s=[45 for _ in robots],
        c=["black" for _ in robots],
        zorder=6,
    )
    labels = [
        ax_world.text(r.x + 0.6, r.y + 0.6, f"R{r.id}", fontsize=8, color="black", zorder=7)
        for r in robots
    ]

    state = {"step": 0, "done": False, "printed": False}

    def update(frame):
        if state["done"]:
            return (world_img, search_img, follow_img, robot_scat, target_plot, *labels)

        for _ in range(steps_per_frame):
            state["step"] += 1
            done = sim_step(
                variant_name=variant_name,
                current_step=state["step"],
                robots=robots,
                advertiser_spirals=advertiser_spirals,
                pher_search=pher_search,
                pher_follow=pher_follow,
                covered=covered,
                targets=targets,
                found_targets=found_targets,
                global_target_visits=global_target_visits,
                convergence_metrics=convergence_metrics,
                tau_decay=tau_decay,
            )
            if done or state["step"] >= max_steps:
                state["done"] = True
                if not state["printed"]:
                    print_metrics(
                        variant_name=variant_name,
                        searchers=robots,
                        convergence_metrics=convergence_metrics,
                        target=target,
                        search_starts=search_starts,
                        requested_distance_shell=requested_distance_shell,
                        final_step=state["step"],
                    )
                    state["printed"] = True
                break

        world_img.set_data(coverage_to_image(covered))
        search_img.set_data(pheromone_to_rgba(pher_search))
        follow_img.set_data(pheromone_to_rgba(pher_follow))
        robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
        colors = [
            "blue" if r.mode == "ADVERTISE" else "green" if target in r.visited_targets else "orange" if r.mode == "FOLLOW" else "black"
            for r in robots
        ]
        robot_scat.set_facecolors(colors)
        for label, r in zip(labels, robots):
            label.set_position((r.x + 0.6, r.y + 0.6))

        distances = [manhattan((searcher.x, searcher.y), target) for searcher in robots]
        n_found = sum(1 for r in robots if target in r.visited_targets)
        ax_world.set_title(
            f"World: step={state['step']}, found={n_found}/{len(robots)}, "
            f"min_d={min(distances)}, mean_d={float(np.mean(distances)):.1f}",
            fontsize=10,
        )
        frame_writer.save(fig)
        return (world_img, search_img, follow_img, robot_scat, target_plot, *labels)

    anim = run_animation(fig, update, frames=max_steps, interval_ms=interval_ms, blit=False)
    plt.show()
    return anim
