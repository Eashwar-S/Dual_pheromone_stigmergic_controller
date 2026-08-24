from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Set, Tuple
import inspect
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# SIM_ROOT = Path(__file__).resolve().parent
# ROBOTARIUM_ROOT = SIM_ROOT.parent
# sys.path.insert(0, str(ROBOTARIUM_ROOT))

import rps.robotarium as robotarium
try:
    from rps.utilities.barrier_certificates import create_si_barrier_certificate
except ImportError:
    from rps.utilities.barrier_certificates import (
        create_single_integrator_barrier_certificate as create_si_barrier_certificate,
    )
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.transformations import create_si_to_uni_mapping

N_ROBOTS = 2
ROBOT_IDS = tuple(range(N_ROBOTS))
GRID_SIZE = 50
W = H = GRID_SIZE

WORLD_BOUNDS = np.array([-1.0, 1.0, -1.0, 1.0])
WORLD_MARGIN = 0.08
ACTIVE_WORLD_BOUNDS = np.array([
    WORLD_BOUNDS[0] + WORLD_MARGIN,
    WORLD_BOUNDS[1] - WORLD_MARGIN,
    WORLD_BOUNDS[2] + WORLD_MARGIN,
    WORLD_BOUNDS[3] - WORLD_MARGIN,
])

TARGET_GRID = (GRID_SIZE // 2, GRID_SIZE // 2)
START_RANDOM_SEED = 14
ALGORITHM_RANDOM_SEED = 131
SEARCH_START_DISTANCE = 30
SEARCH_START_SEPARATION = 10

ROBOT_RADIUS = 2
COLLISION_RADIUS = 4
TARGET_REACHED_RADIUS = ROBOT_RADIUS
PHER_DEPOSIT = 1.0
PHER_MAX = 1.0
PHER_MIN = 1e-6
TAU_DECAY = 2 * (GRID_SIZE ** 2) / (
    N_ROBOTS * max(1, ROBOT_RADIUS)
)

MAX_STEPS = 120000
MAX_WALL_SECONDS = 590.0
GRID_MOVE_CELLS = 7
WAYPOINT_UPDATE_STEPS = 5
SI_PROJECTION_DISTANCE = 0.05
ROBOTARIUM_SAFETY_RADIUS = 0.20
GOAL_SEPARATION_RADIUS = 0.22
STUCK_MOVEMENT_EPS = 0.002
STUCK_STEPS = 80
VELOCITY_LIMIT = 0.10
WAYPOINT_ARRIVAL_CELLS = 1.25

CELL_WIDTH = (
    ACTIVE_WORLD_BOUNDS[1] - ACTIVE_WORLD_BOUNDS[0]
) / W
CELL_HEIGHT = (
    ACTIVE_WORLD_BOUNDS[3] - ACTIVE_WORLD_BOUNDS[2]
) / H


def get_robotarium_axes(r):
    return (
        getattr(r, "_axes_handle", None)
        or getattr(r, "axes_handle", None)
        or getattr(r, "axes", None)
        or getattr(r, "ax", None)
        or (plt.gca() if plt.get_fignums() else None)
    )


def normalize_robotarium_plot_handles(r) -> None:
    ax = get_robotarium_axes(r)
    if ax is not None and not hasattr(r, "_axes_handle"):
        r._axes_handle = ax


def interactive_figure_available() -> bool:
    """Return whether Matplotlib is using a window-capable backend."""
    backend = str(matplotlib.get_backend()).lower()
    non_interactive = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
    }
    return backend not in non_interactive and "inline" not in backend


def resolve_show_figure(show_figure: bool) -> bool:
    if show_figure and not interactive_figure_available():
        print(
            f"Matplotlib backend {matplotlib.get_backend()!r} is non-interactive; "
            "running without a GUI figure. Use an interactive backend or save "
            "the visualization from the script instead."
        )
        return False
    return show_figure


def grid_to_world(gx: int, gy: int) -> Tuple[float, float]:
    wx = (gx + 0.5) * CELL_WIDTH + ACTIVE_WORLD_BOUNDS[0]
    wy = (gy + 0.5) * CELL_HEIGHT + ACTIVE_WORLD_BOUNDS[2]
    return float(wx), float(wy)


def world_to_grid(wx: float, wy: float) -> Tuple[int, int]:
    gx = int((wx - ACTIVE_WORLD_BOUNDS[0]) / CELL_WIDTH)
    gy = int((wy - ACTIVE_WORLD_BOUNDS[2]) / CELL_HEIGHT)
    return (
        int(np.clip(gx, 0, W - 1)),
        int(np.clip(gy, 0, H - 1)),
    )


def clamp_grid_goal(goal: Tuple[int, int]) -> Tuple[int, int]:
    return (
        int(np.clip(goal[0], ROBOT_RADIUS, W - 1 - ROBOT_RADIUS)),
        int(np.clip(goal[1], ROBOT_RADIUS, H - 1 - ROBOT_RADIUS)),
    )


def scale_grid_move(
    start: Tuple[int, int],
    chosen: Tuple[int, int],
    step_size: int = GRID_MOVE_CELLS,
) -> Tuple[int, int]:
    dx = int(np.sign(chosen[0] - start[0]))
    dy = int(np.sign(chosen[1] - start[1]))
    return clamp_grid_goal((
        start[0] + dx * max(1, step_size),
        start[1] + dy * max(1, step_size),
    ))


def seeded_search_starts(
    seed: int = START_RANDOM_SEED,
    manhattan_distance: int = SEARCH_START_DISTANCE,
) -> List[Tuple[int, int]]:
    if manhattan_distance <= TARGET_REACHED_RADIUS:
        raise ValueError("search distance must be outside the target radius")

    rng = np.random.default_rng(seed)
    candidates = [
        (gx, gy)
        for gx in range(ROBOT_RADIUS, W - ROBOT_RADIUS)
        for gy in range(ROBOT_RADIUS, H - ROBOT_RADIUS)
        if abs(gx - TARGET_GRID[0]) + abs(gy - TARGET_GRID[1])
        == manhattan_distance
    ]
    rng.shuffle(candidates)

    starts: List[Tuple[int, int]] = []
    for candidate in candidates:
        if all(
            max(
                abs(candidate[0] - other[0]),
                abs(candidate[1] - other[1]),
            ) >= SEARCH_START_SEPARATION
            for other in starts
        ):
            starts.append(candidate)
        if len(starts) == N_ROBOTS:
            return starts

    raise RuntimeError(
        f"could not place {N_ROBOTS} separated robots exactly "
        f"{manhattan_distance} cells from {TARGET_GRID}"
    )


def initial_grid_positions(
    seed: int = START_RANDOM_SEED,
    search_distance: int = SEARCH_START_DISTANCE,
) -> List[Tuple[int, int]]:
    return seeded_search_starts(seed, search_distance)


def make_initial_conditions(
    grid_positions: Sequence[Tuple[int, int]],
    seed: int = START_RANDOM_SEED,
) -> np.ndarray:
    if len(grid_positions) != N_ROBOTS:
        raise ValueError(f"expected {N_ROBOTS} initial positions")
    rng = np.random.default_rng(seed)
    poses = np.zeros((3, N_ROBOTS), dtype=float)
    for robot_id, (gx, gy) in enumerate(grid_positions):
        poses[0, robot_id], poses[1, robot_id] = grid_to_world(gx, gy)
    poses[2, :] = rng.uniform(-np.pi, np.pi, N_ROBOTS)
    return poses


def create_robotarium(
    show_figure: bool,
    show_grid: bool = True,
    seed: int = START_RANDOM_SEED,
    search_distance: int = SEARCH_START_DISTANCE,
):
    starts = initial_grid_positions(seed, search_distance)
    r = robotarium.Robotarium(
        number_of_robots=N_ROBOTS,
        show_figure=show_figure,
        sim_in_real_time=False,
        initial_conditions=make_initial_conditions(starts, seed),
    )
    normalize_robotarium_plot_handles(r)

    if show_figure:
        ax = get_robotarium_axes(r)
        if ax is not None:
            ax.set_xlim(WORLD_BOUNDS[0], WORLD_BOUNDS[1])
            ax.set_ylim(WORLD_BOUNDS[2], WORLD_BOUNDS[3])
            ax.set_aspect("equal")
            if show_grid:
                x_lines = np.linspace(
                    ACTIVE_WORLD_BOUNDS[0],
                    ACTIVE_WORLD_BOUNDS[1],
                    W + 1,
                )
                y_lines = np.linspace(
                    ACTIVE_WORLD_BOUNDS[2],
                    ACTIVE_WORLD_BOUNDS[3],
                    H + 1,
                )
                ax.vlines(
                    x_lines,
                    ACTIVE_WORLD_BOUNDS[2],
                    ACTIVE_WORLD_BOUNDS[3],
                    color="k",
                    alpha=0.15,
                    linewidth=0.35,
                    zorder=-3,
                )
                ax.hlines(
                    y_lines,
                    ACTIVE_WORLD_BOUNDS[0],
                    ACTIVE_WORLD_BOUNDS[1],
                    color="k",
                    alpha=0.15,
                    linewidth=0.35,
                    zorder=-3,
                )
            ax.add_patch(Rectangle(
                (ACTIVE_WORLD_BOUNDS[0], ACTIVE_WORLD_BOUNDS[2]),
                ACTIVE_WORLD_BOUNDS[1] - ACTIVE_WORLD_BOUNDS[0],
                ACTIVE_WORLD_BOUNDS[3] - ACTIVE_WORLD_BOUNDS[2],
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                zorder=20,
            ))
    return r


def create_motion_helpers():
    barrier_kwargs = {
        "safety_radius": ROBOTARIUM_SAFETY_RADIUS,
        "barrier_gain": 120.0,
        "magnitude_limit": VELOCITY_LIMIT,
    }
    barrier_signature = inspect.signature(create_si_barrier_certificate)
    barrier_kwargs = {
        key: value
        for key, value in barrier_kwargs.items()
        if key in barrier_signature.parameters
    }
    si_barrier = create_si_barrier_certificate(**barrier_kwargs)
    position_controller = create_si_position_controller(
        velocity_magnitude_limit=VELOCITY_LIMIT,
    )
    si_to_uni, uni_to_si = create_si_to_uni_mapping(
        projection_distance=SI_PROJECTION_DISTANCE,
    )
    return si_barrier, position_controller, si_to_uni, uni_to_si


def make_goal_world(
    grid_goals: Sequence[Tuple[int, int]],
) -> np.ndarray:
    goals = np.zeros((2, N_ROBOTS), dtype=float)
    for robot_id, goal in enumerate(grid_goals):
        goals[:, robot_id] = grid_to_world(*clamp_grid_goal(goal))
    return goals


def has_arrived(
    x_si: np.ndarray,
    robot_id: int,
    grid_goal: Tuple[int, int],
    tolerance_cells: float = WAYPOINT_ARRIVAL_CELLS,
) -> bool:
    goal_world = np.array(grid_to_world(*clamp_grid_goal(grid_goal)))
    tolerance = tolerance_cells * min(CELL_WIDTH, CELL_HEIGHT)
    return (
        float(np.linalg.norm(x_si[:, robot_id] - goal_world))
        <= tolerance
    )


def separate_goals(
    goals: np.ndarray,
    x_si: np.ndarray,
    min_distance: float = GOAL_SEPARATION_RADIUS,
) -> np.ndarray:
    adjusted = goals.copy()
    delta = adjusted[:, 0] - adjusted[:, 1]
    distance = float(np.linalg.norm(delta))
    if distance < min_distance:
        if distance < 1e-9:
            delta = x_si[:, 0] - x_si[:, 1]
            distance = float(np.linalg.norm(delta))
        if distance < 1e-9:
            delta = np.array([1.0, 0.0])
            distance = 1.0
        correction = 0.5 * (min_distance - distance) * delta / distance
        adjusted[:, 0] += correction
        adjusted[:, 1] -= correction
    return adjusted


def update_robotarium(r, x_pose, grid_goals, helpers):
    si_barrier, position_controller, si_to_uni, uni_to_si = helpers
    x_si = uni_to_si(x_pose)
    goals = separate_goals(make_goal_world(grid_goals), x_si)
    dxi = position_controller(x_si, goals)
    dxi = si_barrier(dxi, x_si)
    dxu = si_to_uni(dxi, x_pose)
    max_linear_velocity = getattr(r, "MAX_LINEAR_VELOCITY", VELOCITY_LIMIT)
    max_angular_velocity = getattr(r, "MAX_ANGULAR_VELOCITY", 2.0)
    dxu[0, :] = np.clip(
        dxu[0, :],
        -max_linear_velocity,
        max_linear_velocity,
    )
    dxu[1, :] = np.clip(
        dxu[1, :],
        -max_angular_velocity,
        max_angular_velocity,
    )
    r.set_velocities(np.arange(N_ROBOTS), dxu)
    r.step()
    return r.get_poses()


def target_found(
    position: Tuple[int, int],
    radius: int = TARGET_REACHED_RADIUS,
) -> bool:
    return (
        abs(position[0] - TARGET_GRID[0])
        + abs(position[1] - TARGET_GRID[1])
        <= radius
    )


def apply_decay(
    pheromone: np.ndarray,
    tau_decay: float = TAU_DECAY,
    pher_min: float = PHER_MIN,
) -> None:
    pheromone *= np.exp(-1.0 / tau_decay)
    pheromone[pheromone < pher_min] = 0.0
    np.clip(pheromone, 0.0, PHER_MAX, out=pheromone)


def deposit_uniform(
    pheromone: np.ndarray,
    x: int,
    y: int,
    amount: float,
    robot_radius: int,
) -> None:
    """Deposit repulsive pheromone and saturate the field at one."""
    for dy in range(-robot_radius, robot_radius + 1):
        for dx in range(-robot_radius, robot_radius + 1):
            if abs(dx) + abs(dy) > robot_radius:
                continue
            cx, cy = x + dx, y + dy
            if 0 <= cx < pheromone.shape[1] and 0 <= cy < pheromone.shape[0]:
                pheromone[cy, cx] = min(
                    PHER_MAX,
                    pheromone[cy, cx] + amount,
                )


def mark_covered(
    covered: np.ndarray,
    x: int,
    y: int,
    radius: int = ROBOT_RADIUS,
) -> None:
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            cx, cy = x + dx, y + dy
            if 0 <= cx < covered.shape[1] and 0 <= cy < covered.shape[0]:
                covered[cy, cx] = True


def covered_to_rgba(covered: np.ndarray) -> np.ndarray:
    """Dark-blue coverage color used by robotarium_stigmergy_search.py."""
    rgba = np.zeros((*covered.shape, 4), dtype=float)
    covered_cells = covered > 0
    rgba[covered_cells, 0] = 0.00
    rgba[covered_cells, 1] = 0.00
    rgba[covered_cells, 2] = 1.00
    rgba[covered_cells, 3] = 0.88
    return rgba


def pheromone_to_rgba(pheromone: np.ndarray) -> np.ndarray:
    """Pale-blue pheromone color used by robotarium_stigmergy_search.py."""
    rgba = np.zeros((*pheromone.shape, 4), dtype=float)
    rgba[..., 0] = 0.92
    rgba[..., 1] = 0.98
    rgba[..., 2] = 1.00
    rgba[..., 3] = np.clip(pheromone / PHER_MAX, 0.0, 1.0) * 0.75
    return rgba


def add_coverage_overlay(r, covered: np.ndarray, show_figure: bool):
    if not show_figure:
        return None
    return get_robotarium_axes(r).imshow(
        covered_to_rgba(covered),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        interpolation="nearest",
        zorder=-6,
    )


def refresh_coverage_overlay(coverage_plot, covered: np.ndarray) -> None:
    if coverage_plot is not None:
        coverage_plot.set_data(covered_to_rgba(covered))


def add_pheromone_overlay(
    r,
    pheromone: np.ndarray,
    show_figure: bool,
):
    if not show_figure:
        return None
    return get_robotarium_axes(r).imshow(
        pheromone_to_rgba(pheromone),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        interpolation="nearest",
        zorder=-5,
    )


def refresh_pheromone_overlay(pheromone_plot, pheromone: np.ndarray) -> None:
    if pheromone_plot is not None:
        pheromone_plot.set_data(pheromone_to_rgba(pheromone))


def add_target_plot(r, show_figure: bool):
    if not show_figure:
        return None
    ax = get_robotarium_axes(r)
    wx, wy = grid_to_world(*TARGET_GRID)
    ax.scatter(
        [wx],
        [wy],
        s=420,
        marker="x",
        c="black",
        linewidths=6,
        zorder=30,
    )
    ax.scatter(
        [wx],
        [wy],
        s=340,
        marker="x",
        c="darkred",
        linewidths=3.5,
        zorder=31,
    )
    return ax.scatter(
        [],
        [],
        s=430,
        marker="o",
        facecolors="none",
        edgecolors="green",
        linewidths=3,
        zorder=32,
    )


def refresh_target_plot(target_plot, found_robot_ids: Set[int]) -> None:
    if target_plot is not None and found_robot_ids:
        target_plot.set_offsets([grid_to_world(*TARGET_GRID)])


def update_title(
    r,
    step: int,
    found_robot_ids: Set[int],
    show_figure: bool,
) -> None:
    if not show_figure:
        return
    statuses = [
        f"R{robot_id}: SEARCH"
        + (" \N{CHECK MARK}" if robot_id in found_robot_ids else "")
        for robot_id in ROBOT_IDS
    ]
    get_robotarium_axes(r).set_title(
        f"Single-Pheromone Search | Step {step} | "
        f"Found: {len(found_robot_ids)}/{N_ROBOTS} | "
        + " | ".join(statuses)
    )


def add_completion_banner(
    r,
    step: int,
    wall_seconds: float,
    show_figure: bool,
) -> None:
    if not show_figure:
        return
    get_robotarium_axes(r).text(
        0.5,
        0.96,
        f"Both robots found the target at step {step} | "
        f"Wall time {wall_seconds:.2f} s",
        color="red",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="top",
        transform=get_robotarium_axes(r).transAxes,
        zorder=40,
        bbox={
            "facecolor": "white",
            "edgecolor": "red",
            "alpha": 0.85,
            "pad": 5,
        },
    )


def print_position_report(
    step: int,
    grid_positions: Sequence[Tuple[int, int]],
    x_pose: np.ndarray,
    found_robot_ids: Set[int],
    pheromone: np.ndarray,
) -> None:
    parts = []
    for robot_id in ROBOT_IDS:
        status = "found" if robot_id in found_robot_ids else "searching"
        parts.append(
            f"R{robot_id} grid={grid_positions[robot_id]} "
            f"world=({x_pose[0, robot_id]:.3f}, "
            f"{x_pose[1, robot_id]:.3f}) mode=SEARCH {status}"
        )
    print(
        f"Step {step}: "
        + "; ".join(parts)
        + f"; pheromone_max={float(np.max(pheromone)):.4f}"
    )


def should_stop(step: int, start_time: float, max_steps: int) -> bool:
    return (
        step >= max_steps
        or time.perf_counter() - start_time >= MAX_WALL_SECONDS
    )


def finish_robotarium(r) -> None:
    if hasattr(r, "call_at_scripts_end"):
        r.call_at_scripts_end()
    elif hasattr(r, "debug"):
        r.debug()
