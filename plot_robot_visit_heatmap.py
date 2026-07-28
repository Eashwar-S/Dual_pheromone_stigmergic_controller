"""Replay experiment runs listed in a sampled time-series CSV and plot visits.

The sampled CSV contains aggregate statistics rather than robot coordinates.
This script reconstructs centralized, random-walk, and efficient-stigmergy runs
using the deterministic seed conventions in their parallel experiment scripts.

A visit is counted for a robot's starting cell and every cell it subsequently
enters. A failed robot contributes no visits beginning at its failure step, and
a robot that has completed its path contributes no repeated visits while it is
stationary. The final cell entered when completing a path is counted once.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import config_experiments
from centralized.lawnmower_pattern import generate_sweep_path_for_region
from centralized.partitioning import lloyd_balanced
from centralized.robot import Robot
from common.geometry import manhattan_path
from common.utilities import (
    discover_targets_in_vnhood,
    generate_unique_targets,
    mark_visible,
)
from random_walk.robot_random import Robot as RandomWalkRobot
from stigmergy_common.pheromone import apply_decay
from stigmergy_search.robot_efficient import Robot as StigmergyRobot


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = (
    PROJECT_ROOT
    / "centralized"
    / "experiment_results"
    / "E1"
    / "timeseries_sampled.csv"
)
REQUIRED_COLUMNS = {
    "algorithm",
    "grid_size",
    "n_robots",
    "n_targets",
    "n_failures",
    "experiment_id",
}
SUPPORTED_ALGORITHMS = {
    "centralized": "Centralized",
    "random_walk": "Random walk",
    "stigmergy_search_efficient": "Efficient stigmergy",
}
Cell = Tuple[int, int]
Failure = Tuple[int, int]


def _build_paths_and_targets(
    grid_size: int,
    n_robots: int,
    n_targets: int,
    robot_radius: int,
    run_seed: int,
) -> Tuple[List[Robot], Set[Cell]]:
    """Reproduce centralized path planning and target generation for one run."""
    rng = np.random.default_rng(run_seed)
    height = width = grid_size

    yy, xx = np.mgrid[0:height, 0:width]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    labels, centers = lloyd_balanced(
        points,
        n_robots,
        max_iters_centers=10,
        max_iters_assign=30,
        step0=0.1,
        decay=0.1,
        rng=rng,
    )
    zones = labels.reshape(height, width)

    sweep_paths: List[List[Cell]] = []
    for robot_id in range(n_robots):
        path = generate_sweep_path_for_region(
            zones == robot_id,
            robot_radius=robot_radius,
        )
        if not path:
            center_x, center_y = centers[robot_id]
            x = int(np.clip(round(center_x - 0.5), 0, width - 1))
            y = int(np.clip(round(center_y - 0.5), 0, height - 1))
            path = [(x, y)]
        sweep_paths.append(path)

    starts = config_experiments.generate_robot_positions(
        grid_size,
        n_robots,
        rng,
    )
    full_paths: List[List[Cell]] = []
    for start, sweep in zip(starts, sweep_paths):
        navigation = manhattan_path(start, sweep[0])
        full_paths.append(navigation[:-1] + sweep)

    robots = [Robot(robot_id, path) for robot_id, path in enumerate(full_paths)]
    targets = generate_unique_targets(grid_size, n_targets, rng)
    return robots, targets


def _append_failed_path(
    robots: Sequence[Robot],
    takeover_id: int,
    failed_id: int,
) -> None:
    """Apply the same failed-path takeover used by the centralized simulator."""
    takeover = robots[takeover_id]
    failed = robots[failed_id]
    navigation = manhattan_path(takeover.pos, failed.pos)
    remaining = failed.path[failed.idx :]

    extension: List[Cell] = []
    if len(navigation) > 1:
        extension.extend(navigation[1:])
    if len(remaining) > 1:
        extension.extend(remaining[1:])
    if extension:
        takeover.path.extend(extension)

    failed.path = failed.path[: failed.idx + 1]


def _mark_visit(
    visit_counts: np.ndarray,
    x: int,
    y: int,
    visit_radius: int,
) -> None:
    """Count a visit in the robot cell or its Manhattan-radius footprint."""
    mark_visible(visit_counts, x, y, visit_radius)


def replay_visit_counts(
    *,
    grid_size: int,
    n_robots: int,
    n_targets: int,
    failure_schedule: Iterable[Failure],
    run_seed: int,
    robot_radius: int,
    visit_radius: int = 0,
) -> np.ndarray:
    """Replay one centralized run and return its per-cell robot visit counts."""
    robots, targets = _build_paths_and_targets(
        grid_size,
        n_robots,
        n_targets,
        robot_radius,
        run_seed,
    )
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    covered = np.zeros((grid_size, grid_size), dtype=bool)
    visit_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    found_targets: Set[Cell] = set()

    normalized = sorted(
        (
            (int(robot_id), int(step))
            for robot_id, step in failure_schedule
            if 0 <= int(robot_id) < n_robots and int(step) >= 0
        ),
        key=lambda item: (item[1], item[0]),
    )
    failures_by_step: Dict[int, List[int]] = {}
    for robot_id, step in normalized:
        failures_by_step.setdefault(step, []).append(robot_id)
    pending_failures: deque[int] = deque()

    def trigger_failures(step: int) -> None:
        for robot_id in failures_by_step.get(step, []):
            robot = robots[robot_id]
            if not robot.failed:
                robot.failed = True
                pending_failures.append(robot_id)

    # A starting position is one visit, unless the robot has already failed or
    # has no path left to execute.
    trigger_failures(0)
    for robot in robots:
        if not robot.failed and robot.idx < len(robot.path) - 1:
            _mark_visit(visit_counts, robot.x, robot.y, visit_radius)

    steps = 0
    total_cells = grid_size * grid_size
    while steps < max_horizon:
        if steps > 0:
            trigger_failures(steps)

        # Survey from active robots before they move.
        for robot in robots:
            if robot.failed or robot.idx >= len(robot.path) - 1:
                continue
            mark_visible(covered, robot.x, robot.y, robot_radius)
            discover_targets_in_vnhood(
                robot.x,
                robot.y,
                targets,
                found_targets,
                grid_size,
                grid_size,
                robot_radius,
            )

        # Only an actual cell entry is a new visit. The final path cell is
        # counted here once, but it is not counted again on later timesteps.
        for robot in robots:
            if robot.failed or robot.idx >= len(robot.path) - 1:
                continue
            previous_idx = robot.idx
            robot.step(robots)
            if robot.idx != previous_idx:
                _mark_visit(visit_counts, robot.x, robot.y, visit_radius)
                mark_visible(covered, robot.x, robot.y, robot_radius)
                discover_targets_in_vnhood(
                    robot.x,
                    robot.y,
                    targets,
                    found_targets,
                    grid_size,
                    grid_size,
                    robot_radius,
                )

        if pending_failures:
            finishers = [
                robot.id
                for robot in robots
                if not robot.failed and robot.idx >= len(robot.path) - 1
            ]
            for finisher_id in finishers:
                if not pending_failures:
                    break
                _append_failed_path(
                    robots,
                    finisher_id,
                    pending_failures.popleft(),
                )

        steps += 1
        coverage_complete = int(np.count_nonzero(covered)) >= total_cells
        targets_complete = len(found_targets) >= len(targets)
        if coverage_complete and targets_complete:
            break

    return visit_counts


def _failure_map(
    failure_schedule: Iterable[Failure],
    n_robots: int,
) -> Dict[int, List[int]]:
    failures_by_step: Dict[int, List[int]] = {}
    for robot_id, step in failure_schedule:
        robot_id = int(robot_id)
        step = int(step)
        if 0 <= robot_id < n_robots and step >= 0:
            failures_by_step.setdefault(step, []).append(robot_id)
    return failures_by_step


def replay_random_walk_visit_counts(
    *,
    grid_size: int,
    n_robots: int,
    n_targets: int,
    failure_schedule: Iterable[Failure],
    run_seed: int,
    robot_seed: int,
    robot_radius: int,
    visit_radius: int = 0,
) -> np.ndarray:
    """Replay one memoryless random-walk run."""
    rng = np.random.default_rng(run_seed)
    robot_rng = np.random.default_rng(robot_seed)
    starts = config_experiments.generate_robot_positions(
        grid_size,
        n_robots,
        rng,
    )
    robots = [
        RandomWalkRobot(
            id=robot_id,
            x=int(start[0]),
            y=int(start[1]),
            robot_radius=robot_radius,
            collision_radius=1,
            local_covered=np.zeros((grid_size, grid_size), dtype=bool),
        )
        for robot_id, start in enumerate(starts)
    ]
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Cell] = set()
    covered = np.zeros((grid_size, grid_size), dtype=bool)
    visit_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    failures_by_step = _failure_map(failure_schedule, n_robots)
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    total_cells = grid_size * grid_size

    # Handle step-zero failures before counting starting cells.
    for robot_id in failures_by_step.get(0, []):
        robots[robot_id].failed = True
    for robot in robots:
        if not robot.failed:
            _mark_visit(visit_counts, robot.x, robot.y, visit_radius)

    steps = 0
    while steps < max_horizon:
        if steps > 0:
            for robot_id in failures_by_step.get(steps, []):
                robots[robot_id].failed = True

        for robot in robots:
            if robot.failed:
                continue
            mark_visible(
                robot.local_covered,
                robot.x,
                robot.y,
                robot_radius,
            )
            mark_visible(covered, robot.x, robot.y, robot_radius)
            discover_targets_in_vnhood(
                robot.x,
                robot.y,
                targets,
                found_targets,
                grid_size,
                grid_size,
                robot_radius,
            )

        for robot in robots:
            if robot.failed:
                continue
            previous_position = (robot.x, robot.y)
            robot.step(robot_rng, robots)
            if (robot.x, robot.y) != previous_position:
                _mark_visit(
                    visit_counts,
                    robot.x,
                    robot.y,
                    visit_radius,
                )
            mark_visible(
                robot.local_covered,
                robot.x,
                robot.y,
                robot_radius,
            )
            mark_visible(covered, robot.x, robot.y, robot_radius)
            discover_targets_in_vnhood(
                robot.x,
                robot.y,
                targets,
                found_targets,
                grid_size,
                grid_size,
                robot_radius,
            )

        steps += 1
        if (
            int(np.count_nonzero(covered)) >= total_cells
            and len(found_targets) >= len(targets)
        ):
            break

    return visit_counts


def replay_stigmergy_visit_counts(
    *,
    grid_size: int,
    n_robots: int,
    n_targets: int,
    failure_schedule: Iterable[Failure],
    run_seed: int,
    robot_seed: int,
    robot_radius: int,
    visit_radius: int = 0,
) -> np.ndarray:
    """Replay one efficient-stigmergy search run."""
    rng = np.random.default_rng(run_seed)
    random.seed(robot_seed)
    starts = config_experiments.generate_robot_positions(
        grid_size,
        n_robots,
        rng,
    )
    robots = [
        StigmergyRobot(
            id=robot_id,
            x=int(start[0]),
            y=int(start[1]),
            local_covered=np.zeros((grid_size, grid_size), dtype=bool),
        )
        for robot_id, start in enumerate(starts)
    ]
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Cell] = set()
    covered = np.zeros((grid_size, grid_size), dtype=bool)
    visit_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    pheromone = np.zeros((grid_size, grid_size), dtype=float)
    tau_decay = (grid_size**2) / (n_robots * max(1, robot_radius))
    pheromone_min = 1e-6
    pheromone_deposit = 1.0
    collision_radius = 1
    failures_by_step = _failure_map(failure_schedule, n_robots)
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    total_cells = grid_size * grid_size

    # Handle step-zero failures before counting starting cells.
    for robot_id in failures_by_step.get(0, []):
        robots[robot_id].failed = True
    for robot in robots:
        if not robot.failed:
            _mark_visit(visit_counts, robot.x, robot.y, visit_radius)

    steps = 0
    while steps < max_horizon:
        if steps > 0:
            for robot_id in failures_by_step.get(steps, []):
                robots[robot_id].failed = True

        apply_decay(pheromone, tau_decay, pheromone_min)

        for robot in robots:
            if robot.failed:
                continue
            mark_visible(
                robot.local_covered,
                robot.x,
                robot.y,
                robot_radius,
            )
            mark_visible(covered, robot.x, robot.y, robot_radius)
            discover_targets_in_vnhood(
                robot.x,
                robot.y,
                targets,
                found_targets,
                grid_size,
                grid_size,
                robot_radius,
            )
            robot.deposit_pheromone(
                pheromone,
                pheromone_deposit,
                robot_radius,
            )

        for robot in robots:
            if robot.failed:
                continue
            previous_position = (robot.x, robot.y)
            robot.step(
                pheromone,
                robots,
                robot_radius,
                collision_radius,
            )
            if (robot.x, robot.y) != previous_position:
                _mark_visit(
                    visit_counts,
                    robot.x,
                    robot.y,
                    visit_radius,
                )
            mark_visible(
                robot.local_covered,
                robot.x,
                robot.y,
                robot_radius,
            )
            mark_visible(covered, robot.x, robot.y, robot_radius)
            discover_targets_in_vnhood(
                robot.x,
                robot.y,
                targets,
                found_targets,
                grid_size,
                grid_size,
                robot_radius,
            )

        steps += 1
        if (
            int(np.count_nonzero(covered)) >= total_cells
            and len(found_targets) >= len(targets)
        ):
            break

    return visit_counts


def _load_selected_configuration(
    csv_path: Path,
    grid_size: int | None,
    n_robots: int | None,
    algorithm: str | None,
    failure_time_mode: str | None,
) -> Tuple[pd.DataFrame, Dict[str, object], str]:
    frame = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )

    available_algorithms = sorted(
        str(value) for value in frame["algorithm"].dropna().unique()
    )
    if algorithm is None:
        supported_available = [
            value
            for value in available_algorithms
            if value in SUPPORTED_ALGORITHMS
        ]
        if len(supported_available) != 1:
            detail = ", ".join(available_algorithms) or "none"
            raise ValueError(
                "Could not detect one supported algorithm from the CSV "
                f"(found: {detail}); specify --algorithm."
            )
        algorithm = supported_available[0]
    elif algorithm not in available_algorithms:
        raise ValueError(
            f"CSV contains no rows with algorithm={algorithm!r}; "
            f"available values: {available_algorithms}."
        )
    selected = frame.loc[frame["algorithm"] == algorithm].copy()

    if grid_size is None:
        grid_size = int(selected["grid_size"].min())
        print(
            f"No --grid-size supplied; using the smallest available grid ({grid_size})."
        )
    selected = selected.loc[selected["grid_size"] == grid_size]
    if n_robots is not None:
        selected = selected.loc[selected["n_robots"] == n_robots]
    if "failure_time_mode" in selected.columns:
        available_modes = sorted(
            str(value)
            for value in selected["failure_time_mode"].dropna().unique()
        )
        if failure_time_mode is None and len(available_modes) > 1:
            failure_time_mode = (
                "mixed" if "mixed" in available_modes else available_modes[0]
            )
            print(
                "No --failure-time-mode supplied; using "
                f"{failure_time_mode!r}."
            )
        if failure_time_mode is not None:
            if failure_time_mode not in available_modes:
                raise ValueError(
                    f"Failure time mode {failure_time_mode!r} is unavailable; "
                    f"choose one of {available_modes}."
                )
            selected = selected.loc[
                selected["failure_time_mode"] == failure_time_mode
            ]
    if selected.empty:
        qualifier = f"grid_size={grid_size}"
        if n_robots is not None:
            qualifier += f", n_robots={n_robots}"
        raise ValueError(
            f"CSV has no {algorithm!r} configuration with {qualifier}."
        )

    configuration_columns = [
        "grid_size",
        "n_robots",
        "n_targets",
        "n_failures",
    ]
    if "failure_time_mode" in selected.columns:
        configuration_columns.append("failure_time_mode")
    configurations = selected[configuration_columns].drop_duplicates()
    if len(configurations) != 1:
        raise ValueError(
            "Selection matches multiple configurations; also specify --n-robots."
        )

    return selected, configurations.iloc[0].to_dict(), algorithm


def _failure_schedule(
    configuration: Dict[str, object],
    base_seed: int,
    robot_radius: int,
) -> List[Failure]:
    grid_size = int(configuration["grid_size"])
    n_robots = int(configuration["n_robots"])
    n_failures = int(configuration["n_failures"])
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    failure_mode = str(
        configuration.get(
            "failure_time_mode",
            config_experiments.FAILURE_TIME_MODE,
        )
    )
    return config_experiments.make_random_failure_schedule(
        grid_size,
        n_robots,
        n_failures,
        np.random.default_rng(base_seed),
        max_horizon,
        robot_radius=robot_radius,
        failure_time_mode=failure_mode,
    )


def plot_heatmap(
    visit_counts: np.ndarray,
    *,
    title: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    max_visits = int(visit_counts.max())
    white_to_red = LinearSegmentedColormap.from_list(
        "white_to_red",
        ("#ffffff", "#ffb3b3", "#ff0000", "#800000"),
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(
        visit_counts,
        origin="lower",
        interpolation="nearest",
        cmap=white_to_red,
        norm=Normalize(vmin=0, vmax=max(1, max_visits)),
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Robot visit count")
    if max_visits <= 20:
        colorbar.set_ticks(np.arange(0, max(1, max_visits) + 1))
    else:
        colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        colorbar.update_ticks()

    ax.set_title(f"{title}\nMaximum visits in one cell: {max_visits}")
    ax.set_xlabel("Cell x")
    ax.set_ylabel("Cell y")
    ax.set_aspect("equal")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a centralized, random-walk, or efficient-stigmergy "
            "experiment from timeseries_sampled.csv and plot a white-to-red "
            "per-cell robot visit heatmap."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to timeseries_sampled.csv.",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path_option",
        type=Path,
        default=None,
        help=f"Named alternative to the positional CSV path (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Grid size to replay (default: smallest grid in the CSV).",
    )
    parser.add_argument(
        "--n-robots",
        type=int,
        default=None,
        help="Robot count, needed only if a grid size has multiple configurations.",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(SUPPORTED_ALGORITHMS),
        default=None,
        help="Algorithm to replay (default: detect it from the CSV).",
    )
    parser.add_argument(
        "--failure-time-mode",
        choices=sorted(config_experiments.FAILURE_TIME_WINDOWS),
        default=None,
        help=(
            "Failure timing scenario for E2 CSVs (default: mixed when the "
            "CSV contains multiple modes)."
        ),
    )
    experiment_group = parser.add_mutually_exclusive_group()
    experiment_group.add_argument(
        "--experiment-id",
        type=int,
        default=1,
        help="Experiment ID to replay (default: 1).",
    )
    experiment_group.add_argument(
        "--all-experiments",
        action="store_true",
        help="Sum visits from every experiment ID in the selected configuration.",
    )
    simulation_group = parser.add_mutually_exclusive_group()
    simulation_group.add_argument(
        "--simulation-id",
        type=int,
        default=1,
        help=(
            "Stochastic simulation ID for random walk or stigmergy "
            "(default: 1)."
        ),
    )
    simulation_group.add_argument(
        "--all-simulations",
        action="store_true",
        help=(
            "Sum every simulation ID for random walk or stigmergy; may be "
            "combined with --all-experiments."
        ),
    )
    parser.add_argument(
        "--visit-radius",
        type=int,
        default=0,
        help=(
            "Manhattan-radius footprint counted around each robot position "
            "(default: 0, the occupied cell only)."
        ),
    )
    parser.add_argument(
        "--robot-radius",
        type=int,
        default=config_experiments.ROBOT_RADIUS,
        help="Robot sensing/planning radius used by the original run.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=config_experiments.BASE_SEED,
        help="Base seed used to create archived runs (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: beside the input CSV).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open an interactive plot window after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path_argument = args.csv_path_option or args.csv_path or DEFAULT_CSV
    csv_path = csv_path_argument.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if args.visit_radius < 0 or args.robot_radius < 0:
        raise ValueError("--visit-radius and --robot-radius must be non-negative.")

    selected_rows, configuration, algorithm = _load_selected_configuration(
        csv_path,
        args.grid_size,
        args.n_robots,
        args.algorithm,
        args.failure_time_mode,
    )
    available_ids = sorted(
        int(value) for value in selected_rows["experiment_id"].unique()
    )
    if args.all_experiments:
        experiment_ids = available_ids
    else:
        if args.experiment_id not in available_ids:
            raise ValueError(
                f"Experiment ID {args.experiment_id} is unavailable; "
                f"choose one of {available_ids}."
            )
        experiment_ids = [args.experiment_id]

    has_simulations = "simulation_id" in selected_rows.columns
    if has_simulations:
        available_simulation_ids = sorted(
            int(value) for value in selected_rows["simulation_id"].unique()
        )
        if args.all_simulations:
            simulation_ids: List[int | None] = available_simulation_ids
        else:
            if args.simulation_id not in available_simulation_ids:
                raise ValueError(
                    f"Simulation ID {args.simulation_id} is unavailable; "
                    f"choose one of {available_simulation_ids}."
                )
            simulation_ids = [args.simulation_id]
    else:
        if args.all_simulations:
            raise ValueError(
                f"Algorithm {algorithm!r} has no simulation_id dimension."
            )
        simulation_ids = [None]

    grid_size = int(configuration["grid_size"])
    n_robots = int(configuration["n_robots"])
    n_targets = int(configuration["n_targets"])
    schedule = _failure_schedule(
        configuration,
        args.base_seed,
        args.robot_radius,
    )
    combined_counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    for experiment_id in experiment_ids:
        run_seed = args.base_seed + experiment_id
        for simulation_id in simulation_ids:
            simulation_text = (
                ""
                if simulation_id is None
                else f", simulation={simulation_id}"
            )
            print(
                f"Replaying algorithm={algorithm}, grid={grid_size}, "
                f"robots={n_robots}, experiment={experiment_id}"
                f"{simulation_text}..."
            )
            common_arguments = {
                "grid_size": grid_size,
                "n_robots": n_robots,
                "n_targets": n_targets,
                "failure_schedule": schedule,
                "run_seed": run_seed,
                "robot_radius": args.robot_radius,
                "visit_radius": args.visit_radius,
            }
            if algorithm == "centralized":
                counts = replay_visit_counts(**common_arguments)
            else:
                if simulation_id is None:
                    raise RuntimeError(
                        f"Algorithm {algorithm!r} requires simulation_id."
                    )
                stochastic_arguments = {
                    **common_arguments,
                    "robot_seed": run_seed * 1000 + simulation_id,
                }
                if algorithm == "random_walk":
                    counts = replay_random_walk_visit_counts(
                        **stochastic_arguments
                    )
                elif algorithm == "stigmergy_search_efficient":
                    counts = replay_stigmergy_visit_counts(
                        **stochastic_arguments
                    )
                else:
                    raise RuntimeError(f"Unsupported algorithm: {algorithm}")
            combined_counts += counts

    experiment_label = (
        f"all {len(experiment_ids)} experiments"
        if args.all_experiments
        else f"experiment {experiment_ids[0]}"
    )
    simulation_label = ""
    if has_simulations:
        simulation_label = (
            f", all {len(simulation_ids)} simulations"
            if args.all_simulations
            else f", simulation {simulation_ids[0]}"
        )
    output_path = args.output
    if output_path is None:
        experiment_suffix = (
            "all" if args.all_experiments else str(experiment_ids[0])
        )
        simulation_suffix = (
            ""
            if not has_simulations
            else (
                "_simall"
                if args.all_simulations
                else f"_sim{simulation_ids[0]}"
            )
        )
        output_path = csv_path.parent / (
            f"robot_visit_heatmap_{algorithm}_grid{grid_size}_"
            f"exp{experiment_suffix}{simulation_suffix}.png"
        )

    plot_heatmap(
        combined_counts,
        title=(
            f"{SUPPORTED_ALGORITHMS[algorithm]} robot visits — "
            f"{grid_size}×{grid_size}, {n_robots} robots, "
            f"{experiment_label}{simulation_label}"
        ),
        output_path=output_path.expanduser().resolve(),
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
