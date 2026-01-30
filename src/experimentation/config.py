import numpy as np
from typing import List, Tuple
from pathlib import Path


ROBOT_RADIUS = 5
KAPPA = 10.0
BASE_SEED = 42
RUNS_PER_SCENARIO = 5

GRID_SIZES = [50, 100, 150, 200, 250]
ROBOT_COUNTS = [5, 10, 15, 20, 25]
TARGET_COUNTS = [15, 20, 35, 50, 75]
FAILURE_COUNTS = [0, 0, 0, 0, 0, 0]
# FAILURE_COUNTS = [0, 1, 2, 4]

EXPERIMENT_RESULTS_DIR = Path(__file__).parent / "experiment_results"


def calculate_horizon(grid_size: int, n_robots: int, kappa: float = KAPPA) -> int:
    """
    Calculate timeout horizon H = ⌈κ · L²/N⌉
    
    Args:
        grid_size: Grid dimension L (L×L grid)
        n_robots: Number of robots N
        kappa: Slack factor (default 10.0)
    
    Returns:
        Maximum timesteps before timeout
    """
    return int(np.ceil(kappa * (grid_size ** 2) / n_robots))


def make_random_failure_schedule(n_robots: int, n_failures: int, 
                                 rng: np.random.Generator,
                                 max_horizon: int) -> List[Tuple[int, int]]:
    """
    Generate random failure schedule.
    
    Args:
        n_robots: Total number of robots
        n_failures: Number of robots to fail
        rng: Random number generator
        max_horizon: Maximum timestep for failures
    
    Returns:
        List of (robot_id, failure_timestep) tuples
    """
    n_fail = int(min(max(n_failures, 0), n_robots))
    if n_fail == 0:
        return []
    
    robot_ids = rng.choice(n_robots, size=n_fail, replace=False)
    lo = 5
    hi = max(6, int(max_horizon * 0.4))
    steps = rng.integers(low=lo, high=hi, size=n_fail)
    
    return [(int(rid), int(st)) for rid, st in zip(robot_ids, steps)]


def get_experiment_configs():
    """
    Get all experiment parameter combinations.
    
    Returns:
        List of (grid_size, n_robots, n_targets, n_failures) tuples
    """
    configs = []
    for i in range(len(GRID_SIZES)):
        configs.append((
            GRID_SIZES[i],
            ROBOT_COUNTS[i],
            TARGET_COUNTS[i],
            FAILURE_COUNTS[i]
        ))
    return configs


def get_output_path(approach: str) -> Path:
    """
    Get output path for a specific approach.
    
    Args:
        approach: One of 'centralized', 'stigmergy_random', 'stigmergy_rendezvous'
    
    Returns:
        Path to results directory
    """
    path = EXPERIMENT_RESULTS_DIR / approach
    path.mkdir(parents=True, exist_ok=True)
    return path
