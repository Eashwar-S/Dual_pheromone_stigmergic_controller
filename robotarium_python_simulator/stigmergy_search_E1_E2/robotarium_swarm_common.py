from pathlib import Path
from typing import Optional, Set, Tuple, List
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
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from config_experiments import FAILURE_TIME_MODE, make_random_failure_schedule

N_ROBOTS = 5
WORLD_BOUNDS = np.array([-1.0, 1.0, -1.0, 1.0])
WORLD_MARGIN = 0.05
ACTIVE_WORLD_BOUNDS = np.array([
    WORLD_BOUNDS[0] + WORLD_MARGIN,
    WORLD_BOUNDS[1] - WORLD_MARGIN,
    WORLD_BOUNDS[2] + WORLD_MARGIN,
    WORLD_BOUNDS[3] - WORLD_MARGIN,
])

GRID_SIZE = 50
N_TARGETS = 10
RANDOM_SEED = 7
TARGET_RANDOM_SEED = RANDOM_SEED
START_RANDOM_SEED = RANDOM_SEED
ALGORITHM_RANDOM_SEED = RANDOM_SEED + 102
FAILURE_RANDOM_SEED = RANDOM_SEED + 400
SHARED_INITIAL_GRID_POSITIONS = [(1, 22), (17, 24), (34, 25), (1, 1), (25, 1)]

# E2: set this to True to enable robot failures. Leave False for E1/no failures.
E2_FAILURES_ENABLED = True
E2_N_FAILURES = 2

ROBOT_RADIUS = 2
COLLISION_RADIUS = 4
PHER_DEPOSIT = 1.0
TAU_DECAY = 2*(GRID_SIZE ** 2) / (N_ROBOTS * max(1, ROBOT_RADIUS))#600.0
# print(f'Tau decay - {TAU_DECAY}')
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

W = H = GRID_SIZE
CELL_WIDTH = (ACTIVE_WORLD_BOUNDS[1] - ACTIVE_WORLD_BOUNDS[0]) / W
CELL_HEIGHT = (ACTIVE_WORLD_BOUNDS[3] - ACTIVE_WORLD_BOUNDS[2]) / H


def get_robotarium_axes(r):
    return (
        getattr(r, "_axes_handle", None)
        or getattr(r, "axes_handle", None)
        or getattr(r, "axes", None)
        or getattr(r, "ax", None)
        or (plt.gca() if plt.get_fignums() else None)
    )


def get_robotarium_figure(r):
    return (
        getattr(r, "_figure_handle", None)
        or getattr(r, "figure_handle", None)
        or getattr(r, "figure", None)
        or getattr(r, "fig", None)
        or getattr(r, "_fig", None)
        or (plt.gcf() if plt.get_fignums() else None)
    )


def normalize_robotarium_plot_handles(r) -> None:
    ax = get_robotarium_axes(r)
    fig = get_robotarium_figure(r)
    if ax is not None and not hasattr(r, "_axes_handle"):
        r._axes_handle = ax
    if fig is not None and not hasattr(r, "_figure_handle"):
        r._figure_handle = fig

def generate_unique_targets(grid_size: int, m: int, rng: np.random.Generator) -> Set[Tuple[int, int]]:
    """Generate m unique random target positions on a grid."""
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)


def make_target_rng() -> np.random.Generator:
    return np.random.default_rng(TARGET_RANDOM_SEED)


def make_algorithm_rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(ALGORITHM_RANDOM_SEED + offset)


def make_failure_rng() -> np.random.Generator:
    return np.random.default_rng(FAILURE_RANDOM_SEED)


def canonical_targets() -> Set[Tuple[int, int]]:
    return generate_unique_targets(GRID_SIZE, N_TARGETS, make_target_rng())


def make_e2_failure_schedule(enabled: bool = E2_FAILURES_ENABLED,
                             n_failures: int = E2_N_FAILURES,
                             max_horizon: int = MAX_STEPS) -> List[Tuple[int, int]]:
    if not enabled:
        return []
    return make_random_failure_schedule(
        GRID_SIZE,
        N_ROBOTS,
        n_failures,
        make_failure_rng(),
        max_horizon=max_horizon,
        robot_radius=ROBOT_RADIUS,
        failure_time_mode=FAILURE_TIME_MODE,
    )


def active_robot_indices(failed_robot_ids: Set[int]) -> List[int]:
    return [i for i in range(N_ROBOTS) if i not in failed_robot_ids]


def update_failed_robots(step: int, failure_schedule: List[Tuple[int, int]],
                         failed_robot_ids: Set[int], label: str) -> Set[int]:
    for robot_id, failure_step in failure_schedule:
        if step >= failure_step and robot_id not in failed_robot_ids:
            failed_robot_ids.add(robot_id)
            print(f"{label} E2 failure: robot {robot_id} failed at step {step}")
    return failed_robot_ids


def hold_failed_robot_goals(goals: np.ndarray, x_si: np.ndarray,
                            failed_robot_ids: Set[int]) -> np.ndarray:
    if not failed_robot_ids:
        return goals
    adjusted = goals.copy()
    for robot_id in failed_robot_ids:
        adjusted[:, robot_id] = x_si[:, robot_id]
    return adjusted


def should_stop_simulation(step: int, max_steps: int, start_wall_time: float,
                           max_wall_seconds: float = MAX_WALL_SECONDS) -> bool:
    return step >= max_steps or (wall_time() - start_wall_time) >= max_wall_seconds


def print_seed_configuration(label: str, targets: Optional[Set[Tuple[int, int]]] = None,
                             starts: Optional[List[Tuple[int, int]]] = None) -> None:
    print(
        f"{label} seeds: targets={TARGET_RANDOM_SEED}, "
        f"starts={START_RANDOM_SEED}, algorithm={ALGORITHM_RANDOM_SEED}, "
        f"failures={FAILURE_RANDOM_SEED}"
    )
    print(f"{label} starts: {starts if starts is not None else SHARED_INITIAL_GRID_POSITIONS}")
    if targets is not None:
        print(f"{label} targets: {sorted(targets)}")


def print_failure_configuration(label: str, failure_schedule: List[Tuple[int, int]]) -> None:
    if failure_schedule:
        print(f"{label} E2 failures enabled: mode={FAILURE_TIME_MODE}, schedule={failure_schedule}")
    else:
        print(f"{label} E2 failures disabled: schedule=[]")

def neighbors_von_neumann(x: int, y: int, W: int, H: int, r: int) -> List[Tuple[int, int]]:
    """Return all cells within Manhattan distance r from (x, y)."""
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out.append((nx, ny))
    return out

def mark_visible(grid_bool: np.ndarray, x: int, y: int, r: int):
    """Mark all cells within Manhattan distance r from (x, y). Increments int arrays, sets bool arrays to True."""
    H, W = grid_bool.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if grid_bool.dtype == np.int_ or grid_bool.dtype == np.int32 or grid_bool.dtype == np.int64:
                        grid_bool[ny, nx] += 1
                    else:
                        grid_bool[ny, nx] = True


def discover_targets_in_vnhood(x: int, y: int, targets: Set[Tuple[int, int]], 
                                found: Set[Tuple[int, int]], W: int, H: int, r: int):
    """Discover targets within von Neumann neighborhood and add to found set."""
    for (nx, ny) in neighbors_von_neumann(x, y, W, H, r):
        if (nx, ny) in targets:
            found.add((nx, ny))

def world_to_grid(wx: float, wy: float) -> Tuple[int, int]:
    gx = int((wx - ACTIVE_WORLD_BOUNDS[0]) / CELL_WIDTH)
    gy = int((wy - ACTIVE_WORLD_BOUNDS[2]) / CELL_HEIGHT)
    return int(np.clip(gx, 0, W - 1)), int(np.clip(gy, 0, H - 1))


def grid_to_world(gx: int, gy: int) -> Tuple[float, float]:
    wx = (gx + 0.5) * CELL_WIDTH + ACTIVE_WORLD_BOUNDS[0]
    wy = (gy + 0.5) * CELL_HEIGHT + ACTIVE_WORLD_BOUNDS[2]
    return float(wx), float(wy)


def grid_to_safe_world(gx: int, gy: int) -> Tuple[float, float]:
    wx, wy = grid_to_world(gx, gy)
    return wx, wy


def make_initial_conditions(seed: int = START_RANDOM_SEED,
                            grid_positions: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if grid_positions is None:
        grid_positions = SHARED_INITIAL_GRID_POSITIONS
    poses = np.zeros((3, N_ROBOTS), dtype=float)
    for i, (gx, gy) in enumerate(grid_positions[:N_ROBOTS]):
        poses[0, i], poses[1, i] = grid_to_world(gx, gy)
    poses[2, :] = rng.uniform(-np.pi, np.pi, N_ROBOTS)
    return poses


def create_robotarium(show_figure: bool, show_grid: bool = True,
                      initial_grid_positions: Optional[List[Tuple[int, int]]] = None,
                      seed: int = START_RANDOM_SEED):
    r = robotarium.Robotarium(
        number_of_robots=N_ROBOTS,
        show_figure=show_figure,
        sim_in_real_time=False,
        initial_conditions=make_initial_conditions(seed, initial_grid_positions),
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
                ax.vlines(x_lines, ACTIVE_WORLD_BOUNDS[2], ACTIVE_WORLD_BOUNDS[3],
                          color="k", alpha=0.18, linewidth=0.35, zorder=-3)
                ax.hlines(y_lines, ACTIVE_WORLD_BOUNDS[0], ACTIVE_WORLD_BOUNDS[1],
                          color="k", alpha=0.18, linewidth=0.35, zorder=-3)
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
        "barrier_gain": 100.0,
        "magnitude_limit": 0.10,
    }
    barrier_sig = inspect.signature(create_si_barrier_certificate)
    barrier_kwargs = {
        key: value
        for key, value in barrier_kwargs.items()
        if key in barrier_sig.parameters
    }
    si_barrier_cert = create_si_barrier_certificate(**barrier_kwargs)
    si_position_controller = create_si_position_controller(velocity_magnitude_limit=0.10)
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(
        projection_distance=SI_PROJECTION_DISTANCE,
    )
    return si_barrier_cert, si_position_controller, si_to_uni_dyn, uni_to_si_states


def separate_goals(goals: np.ndarray, x_si: np.ndarray,
                   failed_robot_ids: Optional[Set[int]] = None,
                   min_distance: float = GOAL_SEPARATION_RADIUS) -> np.ndarray:
    adjusted = goals.copy()
    failed_robot_ids = failed_robot_ids or set()
    center = np.array([
        0.5 * (ACTIVE_WORLD_BOUNDS[0] + ACTIVE_WORLD_BOUNDS[1]),
        0.5 * (ACTIVE_WORLD_BOUNDS[2] + ACTIVE_WORLD_BOUNDS[3]),
    ])
    for _ in range(3):
        for i in range(N_ROBOTS):
            if i in failed_robot_ids:
                adjusted[:, i] = x_si[:, i]
                continue
            correction = np.zeros(2)
            for j in range(N_ROBOTS):
                if i == j:
                    continue
                obstacle = x_si[:, j] if j in failed_robot_ids else adjusted[:, j]
                delta = adjusted[:, i] - obstacle
                distance = float(np.linalg.norm(delta))
                if distance < 1e-9:
                    delta = adjusted[:, i] - center
                    if float(np.linalg.norm(delta)) < 1e-9:
                        delta = np.array([1.0, 0.0])
                    distance = float(np.linalg.norm(delta))
                if distance < min_distance:
                    correction += (delta / distance) * (min_distance - distance)
            adjusted[:, i] += 0.6 * correction
            adjusted[0, i] = np.clip(adjusted[0, i], ACTIVE_WORLD_BOUNDS[0], ACTIVE_WORLD_BOUNDS[1])
            adjusted[1, i] = np.clip(adjusted[1, i], ACTIVE_WORLD_BOUNDS[2], ACTIVE_WORLD_BOUNDS[3])
    return adjusted


def make_goal_world(robot_grid_positions) -> np.ndarray:
    goals = np.zeros((2, N_ROBOTS), dtype=float)
    for i, (gx, gy) in enumerate(robot_grid_positions):
        goals[:, i] = grid_to_safe_world(gx, gy)
    return goals


def scale_grid_move(start: Tuple[int, int], chosen: Tuple[int, int], step_size: int) -> Tuple[int, int]:
    dx = int(np.sign(chosen[0] - start[0]))
    dy = int(np.sign(chosen[1] - start[1]))
    gx = int(np.clip(start[0] + dx * max(1, step_size), 0, W - 1))
    gy = int(np.clip(start[1] + dy * max(1, step_size), 0, H - 1))
    return gx, gy


def has_arrived(x_si: np.ndarray, robot_id: int, grid_goal: Tuple[int, int],
                tolerance_cells: float = ARRIVAL_TOLERANCE) -> bool:
    wx, wy = grid_to_world(*grid_goal)
    tolerance = tolerance_cells * min(CELL_WIDTH, CELL_HEIGHT)
    return float(np.linalg.norm(x_si[:, robot_id] - np.array([wx, wy]))) <= tolerance


def sense_targets(robot_positions, local_maps, covered_global, targets: Set[Tuple[int, int]],
                  found_targets: Set[Tuple[int, int]], radius: int = ROBOT_RADIUS) -> None:
    for i, (gx, gy) in enumerate(robot_positions):
        mark_visible(local_maps[i], gx, gy, radius)
        mark_visible(covered_global, gx, gy, radius)
        discover_targets_in_vnhood(gx, gy, targets, found_targets, W, H, radius)


def update_robotarium(r, x_pose, x_goal_world, helpers, failed_robot_ids: Optional[Set[int]] = None):
    si_barrier_cert, si_position_controller, si_to_uni_dyn, uni_to_si_states = helpers
    x_si = uni_to_si_states(x_pose)
    x_goal_world = separate_goals(x_goal_world, x_si, failed_robot_ids)
    dxi = si_position_controller(x_si, x_goal_world)
    
    if failed_robot_ids:
        for failed_robot_id in failed_robot_ids:
            dxi[:, failed_robot_id] = 0.0
    
    dxi = si_barrier_cert(dxi, x_si)
    dxu = si_to_uni_dyn(dxi, x_pose)
    
    if failed_robot_ids:
        for robot_id in failed_robot_ids:
            dxu[:, robot_id] = 0.0
    dxu[0, :] = np.clip(dxu[0, :], -r.MAX_LINEAR_VELOCITY, r.MAX_LINEAR_VELOCITY)
    dxu[1, :] = np.clip(dxu[1, :], -r.MAX_ANGULAR_VELOCITY, r.MAX_ANGULAR_VELOCITY)
    r.set_velocities(np.arange(N_ROBOTS), dxu)
    r.step()
    return r.get_poses()


def print_movement_report(label: str, step: int, prev_x_si: np.ndarray,
                          x_si: np.ndarray, interval: int) -> None:
    if interval <= 0 or step % interval != 0:
        return
    movement = np.linalg.norm(x_si - prev_x_si, axis=0)
    print(
        f"{label} step {step}: mean move {movement.mean():.4f} m, "
        f"max move {movement.max():.4f} m, per robot {np.round(movement, 4).tolist()}"
    )


def add_target_plot(r, targets, found_targets, title: str, step: int, show_figure: bool):
    if not show_figure:
        return None
    ax = get_robotarium_axes(r)
    if ax is None:
        return None
    coords = np.array([grid_to_world(gx, gy) for gx, gy in targets]).T
    if coords.size:
        ax.scatter(coords[0], coords[1], s=280, marker="x", c="r",
                   linewidths=2.5, zorder=10)
    found_plot = ax.scatter([], [], s=220, marker="o", facecolors="none",
                            edgecolors="g", linewidths=2.2, zorder=11)
    ax.set_title(f"{title} | Step: {step} | Targets: {len(found_targets)}/{len(targets)}")
    return found_plot


def partition_to_rgba(zones: np.ndarray, alpha: float = 0.22) -> np.ndarray:
    colors = np.array([
        [0.20, 0.47, 0.75, alpha],
        [0.95, 0.50, 0.20, alpha],
        [0.35, 0.70, 0.35, alpha],
        [0.84, 0.37, 0.37, alpha],
        [0.56, 0.44, 0.70, alpha],
    ])
    rgba = colors[np.mod(zones.astype(int), len(colors))]
    return rgba


def add_partition_background(r, zones: np.ndarray, show_figure: bool):
    if not show_figure:
        return None
    return r._axes_handle.imshow(
        partition_to_rgba(zones),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        zorder=-5,
        interpolation="nearest",
    )


def pheromone_to_rgba(pher: np.ndarray, alpha_scale: float = 0.75) -> np.ndarray:
    vmax = max(float(np.percentile(pher, 95)), PHER_MIN)
    norm = np.clip(pher / vmax, 0.0, 1.0)
    rgba = np.zeros((pher.shape[0], pher.shape[1], 4), dtype=float)
    rgba[..., 0] = 0.92
    rgba[..., 1] = 0.98
    rgba[..., 2] = 1.0
    rgba[..., 3] = norm * alpha_scale
    return rgba


def add_pheromone_overlay(r, pher: np.ndarray, show_figure: bool):
    if not show_figure:
        return None
    return r._axes_handle.imshow(
        pheromone_to_rgba(pher),
        origin="lower",
        extent=ACTIVE_WORLD_BOUNDS,
        zorder=-4,
        interpolation="nearest",
    )


def refresh_pheromone_overlay(pheromone_plot, pher: np.ndarray) -> None:
    if pheromone_plot is not None:
        pheromone_plot.set_data(pheromone_to_rgba(pher))


def refresh_found_plot(r, found_plot, found_targets, title: str, step: int, target_count: int) -> None:
    if found_plot is None:
        return
    if found_targets:
        found_coords = np.array([grid_to_world(gx, gy) for gx, gy in found_targets])
        found_plot.set_offsets(found_coords)
    r._axes_handle.set_title(f"{title} | Step: {step} | Targets: {len(found_targets)}/{target_count}")


def add_completion_banner(r, target_step: int, sim_seconds: float, wall_seconds: float,
                          show_figure: bool) -> None:
    if not show_figure:
        return
    r._axes_handle.text(
        0.5,
        0.96,
        f"Targets discovered at step {target_step} | Simulation time {sim_seconds:.2f} s | Wall time {wall_seconds:.2f} s",
        color="red",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="top",
        transform=r._axes_handle.transAxes,
        zorder=30,
        bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.85, "pad": 5},
    )
    if hasattr(r, "_fig"):
        r._fig.canvas.draw_idle()
        r._fig.canvas.flush_events()


def wall_time() -> float:
    return time.perf_counter()


def finish_robotarium(r) -> None:
    if hasattr(r, "call_at_scripts_end"):
        r.call_at_scripts_end()
    elif hasattr(r, "debug"):
        r.debug()

def deposit_uniform(pher: np.ndarray, x: int, y: int, amount: float, robot_radius: int):
    """Deposit uniform pheromone in Manhattan radius r around (x, y)."""
    r = robot_radius
    H, W = pher.shape
    x0, y0 = int(x), int(y)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                cx, cy = x0 + dx, y0 + dy
                if 0 <= cx < W and 0 <= cy < H:
                    pher[cy, cx] += amount


def deposit_distance_signal(pher: np.ndarray, x: int, y: int, target_x: int, target_y: int, robot_radius: int):
    """Deposit exponential distance-based pheromone signal pointing to target."""
    H, W = pher.shape
    r = robot_radius
    x0, y0 = int(x), int(y)
    dist_to_target = np.sqrt((x0 - target_x)**2 + (y0 - target_y)**2)
    signal_strength = 10.0 / (0.1 * dist_to_target + 0.00001)
    
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) <= r:
                cx, cy = x0 + dx, y0 + dy
                if 0 <= cx < W and 0 <= cy < H:
                    pher[cy, cx] = np.maximum(pher[cy, cx], signal_strength)


def apply_decay(pher: np.ndarray, tau_decay: float, pher_min: float = 1e-6):
    """Apply exponential decay to pheromone field."""
    decay_factor = np.exp(-1.0 / tau_decay)
    pher *= decay_factor
    pher[pher < pher_min] = 0.0
