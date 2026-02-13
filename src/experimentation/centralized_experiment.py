from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.geometry import manhattan_path
from src.centralized.partitioning import lloyd_balanced
from src.centralized.path_planning import sensor_aware_path_for_region
from src.centralized.robot import Robot
from src.experimentation import config


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
                   robot_radius: int = 5) -> Dict[str, object]:
    """Run headless centralized simulation and return comprehensive metrics."""
    rng = np.random.default_rng(rng_seed)
    W = H = grid_size
    max_horizon = config.calculate_horizon(grid_size, n_robots)
    
    MAX_ITERS_ASSIGN = 30
    MAX_ITERS_CENTERS = 10
    LAMBDA_STEP0 = 0.1
    LAMBDA_DECAY = 0.1
    
    yy, xx = np.mgrid[0:H, 0:W]
    points = np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))
    labels, centers = lloyd_balanced(points, n_robots, MAX_ITERS_CENTERS, MAX_ITERS_ASSIGN,
                                     LAMBDA_STEP0, LAMBDA_DECAY, rng)
    zones = labels.reshape(H, W)
    
    masks = [(zones == i) for i in range(n_robots)]
    sweeping_paths: List[List[Tuple[int, int]]] = []
    for i in range(n_robots):
        p = sensor_aware_path_for_region(masks[i], robot_radius=robot_radius)
        if not p:
            cx, cy = centers[i]
            x = int(np.clip(round(cx - 0.5), 0, W - 1))
            y = int(np.clip(round(cy - 0.5), 0, H - 1))
            p = [(x, y)]
        sweeping_paths.append(p)
    
    center_pos = (W // 2, H // 2)
    full_paths: List[List[Tuple[int, int]]] = []
    for i in range(n_robots):
        sp = sweeping_paths[i]
        if sp:
            nav = manhattan_path(center_pos, sp[0])
            full_paths.append(nav[:-1] + sp)
        else:
            full_paths.append([center_pos])
    
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
    
    t_targets = max_horizon
    t_coverage = max_horizon
    total_cells = H * W
    
    auc_cov = 0.0
    auc_found = 0.0
    
    steps = 0
    while steps < max_horizon:
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
            r.step()
        
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
    
    return {
        "grid_size": grid_size,
        "n_robots": n_robots,
        "n_targets": len(targets),
        "n_failures": len(norm_sched),
        "failed_robot_ids": ";".join(map(str, [rid for rid, _ in norm_sched])),
        "fail_steps": ";".join(map(str, [st for _, st in norm_sched])),
        "t_targets": t_targets,
        "t_coverage": t_coverage,
        "t_end": t_end,
        "success_targets": success_targets,
        "success_coverage": success_coverage,
        "success_both": success_both,
        "auc_cov": auc_cov,
        "mean_cov": mean_cov,
        "auc_found": auc_found,
        "mean_found": mean_found,
        "pct_revisited": pct_revisited,
    }


def run_experiments():
    """Run all experiments and save results to Excel."""
    output_path = config.get_output_path("centralized")
    xlsx_path = output_path / "results.xlsx"
    
    rows: List[Dict[str, object]] = []
    configs_list = config.get_experiment_configs()
    
    for grid_size, n_robots, n_targets, n_failures in configs_list:
        max_horizon = config.calculate_horizon(grid_size, n_robots)
        
        schedule_seed = (config.BASE_SEED * 10_000) + (grid_size * 100) + (n_robots * 10) + (n_failures * 1000)
        rng_sched = np.random.default_rng(schedule_seed)
        failure_schedule = config.make_random_failure_schedule(n_robots, n_failures, rng_sched, max_horizon)
        
        for run_idx in range(1, config.RUNS_PER_SCENARIO + 1):
            run_seed = schedule_seed #+ run_idx
            
            print(f"Running: grid={grid_size}, robots={n_robots}, targets={n_targets}, failures={n_failures}, run={run_idx}")
            
            result = run_simulation(
                grid_size=grid_size,
                n_robots=n_robots,
                n_targets=n_targets,
                failure_schedule=failure_schedule,
                rng_seed=run_seed,
                robot_radius=config.ROBOT_RADIUS
            )
            
            rows.append(result)
    
    df = pd.DataFrame(rows)
    
    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures"], as_index=False)
          .agg(
              runs=("t_end", "size"),
              avg_t_targets=("t_targets", "mean"),
              avg_t_coverage=("t_coverage", "mean"),
              avg_t_end=("t_end", "mean"),
              sr_targets=("success_targets", "mean"),
              sr_coverage=("success_coverage", "mean"),
              sr_both=("success_both", "mean"),
              avg_auc_cov=("auc_cov", "mean"),
              avg_mean_cov=("mean_cov", "mean"),
              avg_auc_found=("auc_found", "mean"),
              avg_mean_found=("mean_found", "mean"),
              avg_pct_revisited=("pct_revisited", "mean"),
          )
    )
    
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")
        
        for sheet_name, frame in [("detailed", df), ("summary", summary)]:
            ws = writer.sheets[sheet_name]
            for idx, col in enumerate(frame.columns, 1):
                max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
                ws.set_column(idx - 1, idx - 1, min(max_len + 2, 60))
    
    print(f"\nResults saved to: {xlsx_path}")


if __name__ == "__main__":
    run_experiments()
