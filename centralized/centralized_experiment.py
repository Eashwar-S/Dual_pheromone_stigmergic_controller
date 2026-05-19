from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import pandas as pd
import sys
CURRENT_DIR = Path(__file__).resolve().parent          # centralized/
PROJECT_ROOT = CURRENT_DIR.parent                     # stigmergy_new/
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = CURRENT_DIR.parent                     # stigmergy_new/

sys.path.insert(0, str(CURRENT_DIR))

from common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from common.geometry import manhattan_path
from common.communication_tracker import CommunicationTracker
from partitioning import lloyd_balanced
from lawnmower_pattern import generate_sweep_path_for_region
from robot import Robot
import config_experiments


def takeover_append_path(robots: List[Robot], takeover_id: int, failed_id: int):
    """Append failed robot's remaining path to takeover robot."""
    takeover = robots[takeover_id]
    failed = robots[failed_id]
    
    nav = manhattan_path(takeover.pos, failed.pos)
    rem = failed.path[failed.idx:]
    
    extension: List[Tuple[int, int]] = []
    if len(nav) > 1:
        extension += nav[1:]
    if len(rem) > 1:
        extension += rem[1:]
    
    if extension:
        takeover.path.extend(extension)
    
    failed.path = failed.path[:failed.idx + 1]


def run_simulation(grid_size: int, n_robots: int, n_targets: int,
                   failure_schedule: List[Tuple[int, int]], rng_seed: int,
                   robot_radius: int) -> Dict[str, object]:
    """Run headless centralized simulation and return comprehensive metrics."""
    rng = np.random.default_rng(rng_seed)
    W = H = grid_size
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    
    MAX_ITERS_ASSIGN = 30
    MAX_ITERS_CENTERS = 10
    LAMBDA_STEP0 = 0.1
    LAMBDA_DECAY = 0.1
    # COLLISION_RADIUS = 0  # 1 = 3x3 block safe zone
    
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    labels, centers = lloyd_balanced(points, n_robots, MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN,
                                     LAMBDA_STEP0, LAMBDA_DECAY, rng)
    zones = labels.reshape(H, W)
    
    masks = [(zones == i) for i in range(n_robots)]
    sweeping_paths: List[List[Tuple[int, int]]] = []
    for i in range(n_robots):
        p = generate_sweep_path_for_region(masks[i], robot_radius=robot_radius)
        if not p:
            cx, cy = centers[i]
            x = int(np.clip(round(cx - 0.5), 0, W - 1))
            y = int(np.clip(round(cy - 0.5), 0, H - 1))
            p = [(x, y)]
        sweeping_paths.append(p)
    
    # Generate random starting positions for robots
    start_positions = config_experiments.generate_robot_positions(grid_size, n_robots, rng)
    
    # Build full paths: random start -> navigate to sweep path start -> sweep path
    full_paths: List[List[Tuple[int, int]]] = []
    for i in range(n_robots):
        sp = sweeping_paths[i]
        start_pos = start_positions[i]
        if sp:
            nav = manhattan_path(start_pos, sp[0])
            full_paths.append(nav[:-1] + sp)
        else:
            full_paths.append([start_pos])
    
    robots = [Robot(i, full_paths[i]) for i in range(n_robots)]
    
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Tuple[int, int]] = set()
    covered = np.zeros((H, W), dtype=int)
    
    norm_sched: List[Tuple[int, int]] = []
    for rid, st in (failure_schedule or []):
        if 0 <= rid < n_robots and st is not None and st >= 0:
            norm_sched.append((int(rid), int(st)))
    norm_sched.sort(key=lambda x: (x[1], x[0]))
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)
    
    from collections import deque
    pending_failures = deque()
    
    # Initialize communication tracker
    # comm_tracker = CommunicationTracker(message_size_bytes=5)
    
    t_targets = max_horizon
    t_coverage = max_horizon
    total_cells = H * W
    
    auc_cov = 0.0
    auc_found = 0.0
    
    steps = 0
    while steps < max_horizon:
        # Track if any failures were reallocated this step
        reallocations_this_step = 0
        
        if steps in fail_map:
            for rid in fail_map[steps]:
                r = robots[rid]
                if not r.failed:
                    r.failed = True
                    pending_failures.append(rid)
        
        for r in robots:
            x, y = r.pos
            mark_visible(covered, x, y, robot_radius)
            discover_targets_in_vnhood(x, y, targets, found_targets, W, H, robot_radius)
        
        for r in robots:
            r.step(robots)
        
        for r in robots:
            x, y = r.pos
            mark_visible(covered, x, y, robot_radius)
            discover_targets_in_vnhood(x, y, targets, found_targets, W, H, robot_radius)
        
        if pending_failures:
            finishers = [rr.id for rr in robots if (not rr.failed) and (rr.idx >= len(rr.path) - 1)]
            for fin_id in finishers:
                if not pending_failures:
                    break
                failed_id = pending_failures.popleft()
                takeover_append_path(robots, fin_id, failed_id)
                reallocations_this_step += 1  # Count reallocation for communication tracking
        
        # Track communication: regular messages + replanning messages
        # active_robots = sum(1 for r in robots if not r.failed)
        # messages_this_step = 2 * active_robots  # Waypoints + acknowledgments
        # messages_this_step += reallocations_this_step  # Replanned waypoints to takeover robots
        # comm_tracker.record_step(messages_this_step)
        
        steps += 1
        
        current_coverage = np.sum(covered > 0)
        coverage_fraction = current_coverage / total_cells
        targets_fraction = len(found_targets) / len(targets) if len(targets) > 0 else 1.0
        
        auc_cov += coverage_fraction
        auc_found += targets_fraction
        
        if len(found_targets) >= len(targets) and t_targets == max_horizon:
            t_targets = steps
        
        if current_coverage >= total_cells and t_coverage == max_horizon:
            t_coverage = steps
        
        if t_targets < max_horizon and t_coverage < max_horizon:
            break
    
    t_end = steps
    mean_cov = auc_cov / t_end if t_end > 0 else 0.0
    mean_found = auc_found / t_end if t_end > 0 else 0.0
    
    success_targets = 1 if t_targets < max_horizon else 0
    success_coverage = 1 if t_coverage < max_horizon else 0
    success_both = 1 if (success_targets and success_coverage) else 0
    
    revisited = np.sum(covered > 1)
    pct_revisited = (revisited / total_cells) * 100.0
    
    # Get communication metrics
    # comm_metrics = comm_tracker.get_metrics()
    
    current_coverage = np.sum(covered > 0)
    percent_coverage = (current_coverage / total_cells) * 100.0

    return {
        "grid_size": grid_size,
        "n_robots": n_robots,
        "n_targets": len(targets),
        "n_failures": len(norm_sched),
        # "failed_robot_ids": ";".join(map(str, [rid for rid, _ in norm_sched])),
        # "fail_steps": ";".join(map(str, [st for _, st in norm_sched])),
        "n_targets_found": len(found_targets),
        "t_targets": t_targets,
        "t_coverage": t_coverage,
        "percent_coverage": percent_coverage,
        "mean_found": mean_found,
        # "t_end": t_end,
        # "success_targets": success_targets,
        # "success_coverage": success_coverage,
        # "success_both": success_both,
        # "auc_cov": auc_cov,
        # "mean_cov": mean_cov,
        # "auc_found": auc_found,
        # "pct_revisited": pct_revisited,
        # # Communication bandwidth metrics
        # "total_messages": comm_metrics['total_messages'],
        # "total_bandwidth_bytes": comm_metrics['total_bandwidth_bytes'],
        # "peak_messages_per_step": comm_metrics['peak_messages_per_step'],
        # "peak_bandwidth_bytes": comm_metrics['peak_bandwidth_bytes'],
        # "avg_messages_per_step": comm_metrics['avg_messages_per_step'],
    }


def run_experiments():
    """Run all centralized experiments and save results to Excel."""
    output_path = Path.cwd() / "experiment_results"
    output_path.mkdir(exist_ok=True)
    NUM_EXPERIMENTS = 10    
    
    rows: List[Dict[str, object]] = []
    configs_list = config_experiments.get_experiment_configs()
    is_failure_experiment = any(cfg[3] > 0 for cfg in configs_list)
    failure_time_modes = list(config_experiments.FAILURE_TIME_WINDOWS) if is_failure_experiment else [config_experiments.FAILURE_TIME_MODE]

    # Determine save path based on first config
    if is_failure_experiment:
        xlsx_path = output_path / "E2/results.xlsx"
        dir_path = output_path / "E2"
    else:
        xlsx_path = output_path / "E1/results.xlsx"
        dir_path = output_path / "E1"
    dir_path.mkdir(exist_ok=True)

    for grid_size, n_robots, n_targets, n_failures in configs_list:
        max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
        modes_for_config = failure_time_modes if n_failures > 0 else [config_experiments.FAILURE_TIME_MODE]

        for failure_time_mode in modes_for_config:
            schedule_seed = 42
            rng_sched = np.random.default_rng(schedule_seed)
            failure_schedule = config_experiments.make_random_failure_schedule(
                grid_size=grid_size,
                n_robots=n_robots,
                n_failures=n_failures,
                rng=rng_sched,
                max_horizon=max_horizon,
                failure_time_mode=failure_time_mode,
            )
            
            for exp_idx in range(1, NUM_EXPERIMENTS + 1): 
                run_seed = schedule_seed + exp_idx
                
                print(f"Running Centralized: grid={grid_size}, robots={n_robots}, targets={n_targets}, failures={n_failures}, failure_time_mode={failure_time_mode}, exp={exp_idx}")
                
                result = run_simulation(
                    grid_size=grid_size,
                    n_robots=n_robots,
                    n_targets=n_targets,
                    failure_schedule=failure_schedule,
                    rng_seed=run_seed,
                    robot_radius=config_experiments.ROBOT_RADIUS
                )
                
                # Inject IDs and constants
                result["failure_time_mode"] = failure_time_mode
                result["experiment_id"] = exp_idx
                result["num_experiments"] = NUM_EXPERIMENTS
                
                print(f't_targets: {result["t_targets"]}, t_coverage: {result["t_coverage"]}, percent_coverage: {result["percent_coverage"]:.2f}%, mean_found: {result["mean_found"]:.4f}')
                
                rows.append(result)
    
    df = pd.DataFrame(rows)
    
    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures", "failure_time_mode"], as_index=False)
          .agg(
              total_runs=("t_targets", "size"),
              avg_n_targets_found=("n_targets_found", "mean"),
              avg_t_targets=("t_targets", "mean"),
              avg_t_coverage=("t_coverage", "mean"),
              avg_percent_coverage=("percent_coverage", "mean"),
              avg_mean_found=("mean_found", "mean"),
          )
    )
    
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        # Reorder columns to make identifying variables lead the sheet
        cols = df.columns.tolist()
        lead_cols = ['grid_size', 'n_robots', 'n_targets', 'n_failures', 'failure_time_mode', 'num_experiments', 'experiment_id']
        cols = lead_cols + [c for c in cols if c not in lead_cols]
        df = df[cols]

        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")
        
        for sheet_name, frame in [("detailed", df), ("summary", summary)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx - 1, idx - 1, min(max_len + 2, 60))
    
    print(f"\nResults saved to: {xlsx_path}")


if __name__ == "__main__":
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization instead of headless experiments"
    )

    args = parser.parse_args()

    if args.visualize:
        visualization_script = CURRENT_DIR / "centralized_search.py"
        subprocess.run([sys.executable, str(visualization_script)])
    else:
        run_experiments()
