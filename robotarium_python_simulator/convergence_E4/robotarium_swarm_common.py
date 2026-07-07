from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import inspect
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# SIM_ROOT = Path(__file__).resolve().parent
# PROJECT_ROOT = SIM_ROOT.parent
# REPO_ROOT = PROJECT_ROOT.parent
# sys.path.insert(0, str(SIM_ROOT))
# sys.path.insert(0, str(PROJECT_ROOT))
# sys.path.insert(0, str(REPO_ROOT))

import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import create_si_barrier_certificate
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.transformations import create_si_to_uni_mapping

N_SEARCH_ROBOTS = 1
N_ROBOTS = 1 + N_SEARCH_ROBOTS
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
ADVERTISER_ID = 0
SEARCH_IDS = tuple(range(1, N_ROBOTS))

START_RANDOM_SEED = 9
ALGORITHM_RANDOM_SEED = 131
SEARCH_START_DISTANCE = 30
SEARCH_START_SEPARATION = 7

ROBOT_RADIUS = 2
COLLISION_RADIUS = 4
PHER_DEPOSIT = 1.0
TAU_DECAY = 2 * (GRID_SIZE ** 2) / (N_ROBOTS * max(1, ROBOT_RADIUS))
PHER_MIN = 1e-6
MAX_STEPS = 120000
MAX_WALL_SECONDS = 590.0
ARRIVAL_TOLERANCE = 0.65
GRID_MOVE_CELLS = 7
WAYPOINT_UPDATE_STEPS = 5
SI_PROJECTION_DISTANCE = 0.05
ROBOTARIUM_SAFETY_RADIUS = 0.20
GOAL_SEPARATION_RADIUS = 0.22
STUCK_MOVEMENT_EPS = 0.002
STUCK_STEPS = 80

ATTRACTIVE_RADIUS = 3
ATTRACTIVE_SENSING_RADIUS = 6
SPIRAL_LANE_SPACING = 4
TARGET_REACHED_RADIUS = 2
TARGET_PARKING_OFFSETS = (
    (0, 8),
    (8, 0),
    (0, -8),
    (-8, 0),
    (6, 6),
    (6, -6),
    (-6, -6),
    (-6, 6),
)

VELOCITY_LIMIT = 0.10

CELL_WIDTH = (ACTIVE_WORLD_BOUNDS[1] - ACTIVE_WORLD_BOUNDS[0]) / W
CELL_HEIGHT = (ACTIVE_WORLD_BOUNDS[3] - ACTIVE_WORLD_BOUNDS[2]) / H


def configure_search_robot_count(count: int) -> None:
    global N_SEARCH_ROBOTS, N_ROBOTS, SEARCH_IDS, TAU_DECAY
    if not 1 <= count <= 3:
        raise ValueError("search robot count must be between 1 and 3")
    N_SEARCH_ROBOTS = count
    N_ROBOTS = 1 + count
    SEARCH_IDS = tuple(range(1, N_ROBOTS))
    TAU_DECAY = 2 * (GRID_SIZE ** 2) / (N_ROBOTS * max(1, ROBOT_RADIUS))


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


def grid_to_world(gx: int, gy: int) -> Tuple[float, float]:
    wx = (gx + 0.5) * CELL_WIDTH + ACTIVE_WORLD_BOUNDS[0]
    wy = (gy + 0.5) * CELL_HEIGHT + ACTIVE_WORLD_BOUNDS[2]
    return float(wx), float(wy)


def world_to_grid(wx: float, wy: float) -> Tuple[int, int]:
    gx = int((wx - ACTIVE_WORLD_BOUNDS[0]) / CELL_WIDTH)
    gy = int((wy - ACTIVE_WORLD_BOUNDS[2]) / CELL_HEIGHT)
    return int(np.clip(gx, 0, W - 1)), int(np.clip(gy, 0, H - 1))


def clamp_grid_goal(goal: Tuple[int, int]) -> Tuple[int, int]:
    margin = ROBOT_RADIUS
    return (
        int(np.clip(goal[0], margin, W - 1 - margin)),
        int(np.clip(goal[1], margin, H - 1 - margin)),
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
    if manhattan_distance <= 0:
        raise ValueError("search start Manhattan distance must be positive")

    rng = np.random.default_rng(seed)
    candidates = [
        (gx, gy)
        for gx in range(ROBOT_RADIUS, W - ROBOT_RADIUS)
        for gy in range(ROBOT_RADIUS, H - ROBOT_RADIUS)
        if abs(gx - TARGET_GRID[0]) + abs(gy - TARGET_GRID[1])
        == manhattan_distance
    ]
    if not candidates:
        raise ValueError(
            f"no valid cells are exactly {manhattan_distance} Manhattan cells "
            f"from target {TARGET_GRID}"
        )

    rng.shuffle(candidates)
    starts: List[Tuple[int, int]] = []
    for gx, gy in candidates:
        separated = all(
            max(abs(gx - sx), abs(gy - sy)) >= SEARCH_START_SEPARATION
            for sx, sy in starts
        )
        if separated:
            starts.append((gx, gy))
        if len(starts) == N_SEARCH_ROBOTS:
            return starts

    raise RuntimeError(
        f"could not place {N_SEARCH_ROBOTS} separated search robots exactly "
        f"{manhattan_distance} Manhattan cells from the target"
    )


def initial_grid_positions(
    seed: int = START_RANDOM_SEED,
    search_distance: int = SEARCH_START_DISTANCE,
) -> List[Tuple[int, int]]:
    return [TARGET_GRID, *seeded_search_starts(seed, search_distance)]


def make_initial_conditions(
    grid_positions: Sequence[Tuple[int, int]],
    seed: int = START_RANDOM_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    poses = np.zeros((3, N_ROBOTS), dtype=float)
    for i, (gx, gy) in enumerate(grid_positions):
        poses[0, i], poses[1, i] = grid_to_world(gx, gy)
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
                x_lines = np.linspace(ACTIVE_WORLD_BOUNDS[0], ACTIVE_WORLD_BOUNDS[1], W + 1)
                y_lines = np.linspace(ACTIVE_WORLD_BOUNDS[2], ACTIVE_WORLD_BOUNDS[3], H + 1)
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
    barrier_sig = inspect.signature(create_si_barrier_certificate)
    barrier_kwargs = {
        key: value for key, value in barrier_kwargs.items()
        if key in barrier_sig.parameters
    }
    si_barrier = create_si_barrier_certificate(**barrier_kwargs)
    position_controller = create_si_position_controller(
        velocity_magnitude_limit=VELOCITY_LIMIT,
    )
    si_to_uni, uni_to_si = create_si_to_uni_mapping(
        projection_distance=SI_PROJECTION_DISTANCE,
    )
    return si_barrier, position_controller, si_to_uni, uni_to_si


def make_goal_world(grid_goals: Sequence[Tuple[int, int]]) -> np.ndarray:
    goals = np.zeros((2, N_ROBOTS), dtype=float)
    for i, goal in enumerate(grid_goals):
        goals[:, i] = grid_to_world(*clamp_grid_goal(goal))
    return goals


def separate_goals(
    goals: np.ndarray,
    x_si: np.ndarray,
    min_distance: float = GOAL_SEPARATION_RADIUS,
) -> np.ndarray:
    adjusted = goals.copy()
    for _ in range(3):
        for i in range(N_ROBOTS - 1):
            for j in range(i + 1, N_ROBOTS):
                delta = adjusted[:, i] - adjusted[:, j]
                distance = float(np.linalg.norm(delta))
                if distance >= min_distance:
                    continue
                if distance < 1e-9:
                    delta = x_si[:, i] - x_si[:, j]
                    distance = float(np.linalg.norm(delta))
                if distance < 1e-9:
                    delta = np.array([1.0, 0.0])
                    distance = 1.0
                correction = 0.5 * (min_distance - distance) * delta / distance
                adjusted[:, i] += correction
                adjusted[:, j] -= correction
    return adjusted


def update_robotarium(r, x_pose, grid_goals, helpers, held_robot_ids=()):
    si_barrier, position_controller, si_to_uni, uni_to_si = helpers
    x_si = uni_to_si(x_pose)
    goals = separate_goals(make_goal_world(grid_goals), x_si)
    dxi = position_controller(x_si, goals)
    for robot_id in held_robot_ids:
        dxi[:, robot_id] = 0.0
    dxi = si_barrier(dxi, x_si)
    dxu = si_to_uni(dxi, x_pose)
    for robot_id in held_robot_ids:
        dxu[:, robot_id] = 0.0
    dxu[0, :] = np.clip(dxu[0, :], -r.MAX_LINEAR_VELOCITY, r.MAX_LINEAR_VELOCITY)
    dxu[1, :] = np.clip(dxu[1, :], -r.MAX_ANGULAR_VELOCITY, r.MAX_ANGULAR_VELOCITY)
    r.set_velocities(np.arange(N_ROBOTS), dxu)
    r.step()
    return r.get_poses()


def has_arrived(
    x_si: np.ndarray,
    robot_id: int,
    grid_goal: Tuple[int, int],
    tolerance_cells: float = ARRIVAL_TOLERANCE,
) -> bool:
    goal = np.array(grid_to_world(*grid_goal))
    tolerance = tolerance_cells * min(CELL_WIDTH, CELL_HEIGHT)
    return float(np.linalg.norm(x_si[:, robot_id] - goal)) <= tolerance


def target_found(
    position: Tuple[int, int],
    radius: int = TARGET_REACHED_RADIUS,
) -> bool:
    return abs(position[0] - TARGET_GRID[0]) + abs(position[1] - TARGET_GRID[1]) <= radius


def apply_decay(pher: np.ndarray, tau_decay: float, pher_min: float = PHER_MIN) -> None:
    pher *= np.exp(-1.0 / tau_decay)
    pher[pher < pher_min] = 0.0


def deposit_uniform(
    pher: np.ndarray,
    x: int,
    y: int,
    amount: float,
    robot_radius: int,
) -> None:
    for dy in range(-robot_radius, robot_radius + 1):
        for dx in range(-robot_radius, robot_radius + 1):
            if abs(dx) + abs(dy) <= robot_radius:
                cx, cy = x + dx, y + dy
                if 0 <= cx < pher.shape[1] and 0 <= cy < pher.shape[0]:
                    pher[cy, cx] += amount


def deposit_distance_signal(
    pher: np.ndarray,
    x: int,
    y: int,
    target_x: int,
    target_y: int,
    robot_radius: int,
) -> None:
    distance = float(np.hypot(x - target_x, y - target_y))
    signal = 1.0 / (1.0 + 0.18 * distance)
    for dy in range(-robot_radius, robot_radius + 1):
        for dx in range(-robot_radius, robot_radius + 1):
            if abs(dx) + abs(dy) <= robot_radius:
                cx, cy = x + dx, y + dy
                if 0 <= cx < pher.shape[1] and 0 <= cy < pher.shape[0]:
                    pher[cy, cx] = max(pher[cy, cx], signal)


def mark_covered(
    covered: np.ndarray,
    x: int,
    y: int,
    radius: int = ROBOT_RADIUS,
) -> None:
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dx) + abs(dy) <= radius:
                cx, cy = x + dx, y + dy
                if 0 <= cx < covered.shape[1] and 0 <= cy < covered.shape[0]:
                    covered[cy, cx] = True


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


def _pheromone_rgba(
    pher: np.ndarray,
    color: Tuple[float, float, float],
    alpha_scale: float,
) -> np.ndarray:
    vmax = max(float(np.max(pher)), PHER_MIN)
    norm = np.clip(pher / vmax, 0.0, 1.0)
    rgba = np.zeros((*pher.shape, 4), dtype=float)
    rgba[..., :3] = color
    rgba[..., 3] = norm * alpha_scale
    return rgba


def blue_pheromone_rgba(pher: np.ndarray) -> np.ndarray:
    vmax = max(float(np.percentile(pher, 95)), PHER_MIN)
    norm = np.clip(pher / vmax, 0.0, 1.0)
    rgba = np.zeros((pher.shape[0], pher.shape[1], 4), dtype=float)
    rgba[..., 0] = 0.92
    rgba[..., 1] = 0.98
    rgba[..., 2] = 1.0
    rgba[..., 3] = norm * 0.75
    return rgba


def pink_pheromone_rgba(pher: np.ndarray) -> np.ndarray:
    return _pheromone_rgba(pher, (1.0, 0.12, 0.55), 0.82)


def add_pheromone_overlays(r, blue: np.ndarray, pink: np.ndarray, show_figure: bool):
    if not show_figure:
        return None, None
    ax = get_robotarium_axes(r)
    blue_plot = ax.imshow(
        blue_pheromone_rgba(blue),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        interpolation="nearest",
        zorder=-5,
    )
    pink_plot = ax.imshow(
        pink_pheromone_rgba(pink),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        interpolation="nearest",
        zorder=-4,
    )
    return blue_plot, pink_plot


def refresh_pheromone_overlays(plots, blue: np.ndarray, pink: np.ndarray) -> None:
    blue_plot, pink_plot = plots
    if blue_plot is not None:
        blue_plot.set_data(blue_pheromone_rgba(blue))
    if pink_plot is not None:
        pink_plot.set_data(pink_pheromone_rgba(pink))


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
    return ax.scatter(
        [wx],
        [wy],
        s=340,
        marker="x",
        c="darkred",
        linewidths=3.5,
        zorder=31,
    )


def update_title(r, step: int, modes: Sequence[str], show_figure: bool) -> None:
    if show_figure:
        found_count = sum(mode == "FOUND" for mode in modes)
        get_robotarium_axes(r).set_title(
            f"Robotarium Rendezvous | Step {step} | "
            f"Found: {found_count}/{len(modes)} | {', '.join(modes)}"
        )


def print_position_report(
    step: int,
    grid_positions: Sequence[Tuple[int, int]],
    x_pose: np.ndarray,
    modes: Sequence[str],
) -> None:
    parts = [
        f"Step {step}: advertiser grid={grid_positions[ADVERTISER_ID]} "
        f"world=({x_pose[0, ADVERTISER_ID]:.3f}, {x_pose[1, ADVERTISER_ID]:.3f})"
    ]
    for mode, robot_id in zip(modes, SEARCH_IDS):
        parts.append(
            f"search-{robot_id} grid={grid_positions[robot_id]} "
            f"world=({x_pose[0, robot_id]:.3f}, {x_pose[1, robot_id]:.3f}) "
            f"mode={mode}"
        )
    print("; ".join(parts))


def should_stop(step: int, start_time: float, max_steps: int) -> bool:
    return step >= max_steps or time.perf_counter() - start_time >= MAX_WALL_SECONDS


def finish_robotarium(r) -> None:
    if hasattr(r, "call_at_scripts_end"):
        r.call_at_scripts_end()
    elif hasattr(r, "debug"):
        r.debug()
