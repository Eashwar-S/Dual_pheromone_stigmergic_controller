from pathlib import Path
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
import sys
import argparse
import subprocess

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from stigmergy_common.pheromone import apply_decay
from robot_random import Robot
import config_experiments


def run_simulation(grid_size: int, n_robots: int, n_targets: int,
                   failure_schedule: List[Tuple[int, int]], rng_seed: int,
                   robot_seed:int,
                   robot_radius: int) -> Dict[str, object]:
    """Run headless stigmergy random walk simulation."""
    rng = np.random.default_rng(rng_seed)
    robot_rng = np.random.default_rng(robot_seed)
    W = H = grid_size
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    
    start_positions = config_experiments.generate_robot_positions(grid_size, n_robots, rng)
    robots = [Robot(id=i, x=start_positions[i][0], y=start_positions[i][1],
                    robot_radius=robot_radius, collision_radius=1,
                    local_covered=np.zeros((H, W), dtype=bool)) for i in range(n_robots)]
    
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Tuple[int, int]] = set()
    covered_global = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)
    
    tau_decay = (grid_size ** 2) / (n_robots * max(1, robot_radius))
    pher_min, pher_deposit = 1e-6, 1.0
    
    norm_sched = [(int(rid), int(st)) for rid, st in (failure_schedule or []) if 0 <= rid < n_robots and st is not None and st >= 0]
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)
        
    t_targets, t_coverage = max_horizon, max_horizon
    total_cells = H * W
    auc_cov, auc_found = 0.0, 0.0
    steps = 0
    
    while steps < max_horizon:
        if steps in fail_map:
            for rid in fail_map[steps]:
                robots[rid].failed = True
                
        apply_decay(pher, tau_decay, pher_min)
        
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered_global, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
                r.deposit_pheromone(pher, pher_deposit)
                
        for r in robots:
            if not r.failed:
                r.step(pher, robot_rng, robots)
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered_global, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
                
        steps += 1
        
        current_coverage = np.sum(covered_global)
        auc_cov += current_coverage / total_cells
        auc_found += len(found_targets) / len(targets) if len(targets) > 0 else 1.0
        
        if len(found_targets) >= len(targets) and t_targets == max_horizon:
            t_targets = steps
        if current_coverage >= total_cells and t_coverage == max_horizon:
            t_coverage = steps
            
        if t_targets < max_horizon and t_coverage < max_horizon:
            break

    mean_found = auc_found / steps if steps > 0 else 0.0
    percent_coverage = (np.sum(covered_global) / total_cells) * 100.0

    return {
        "grid_size": grid_size,
        "n_robots": n_robots,
        "n_targets": len(targets),
        "n_failures": len(norm_sched),
        "n_targets_found": len(found_targets),
        "t_targets": t_targets,
        "t_coverage": t_coverage,
        "percent_coverage": percent_coverage,
        "mean_found": mean_found,
    }


def run_experiments():
    """Run all experiments and save results to Excel."""
    output_path = Path.cwd() / "experiment_results_stigmergy"
    output_path.mkdir(exist_ok=True)
    NUM_EXPERIMENTS = 10
    NUM_SIMULATION = 10
    
    rows: List[Dict[str, object]] = []
    configs_list = config_experiments.get_experiment_configs()

    # Determine save path based on first config
    if configs_list and configs_list[0][3] > 0:
        xlsx_path = output_path / "E2/results.xlsx"
        dir_path = output_path / "E2"
    else:
        xlsx_path = output_path / "E1/results.xlsx"
        dir_path = output_path / "E1"
    dir_path.mkdir(exist_ok=True)

    for grid_size, n_robots, n_targets, n_failures in configs_list:
        max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
        schedule_seed = 42 
        rng_sched = np.random.default_rng(schedule_seed)
        failure_schedule = config_experiments.make_random_failure_schedule(
            grid_size,
            n_robots,
            n_failures,
            rng_sched,
            max_horizon,
        )
        
        # Tier 1: The Environment Layout (M)
        for exp_idx in range(1, NUM_EXPERIMENTS + 1): 
            run_seed = schedule_seed + exp_idx
            
            # Tier 2: The Stochastic Action Trial (N)
            for sim_idx in range(1, NUM_SIMULATION + 1):
                robot_seed = run_seed * 1000 + sim_idx 
                
                print(f"Running Stigmergy: grid={grid_size}, robots={n_robots}, targets={n_targets}, failures={n_failures}, exp={exp_idx}, sim={sim_idx}")
                
                result = run_simulation(
                    grid_size=grid_size,
                    n_robots=n_robots,
                    n_targets=n_targets,
                    failure_schedule=failure_schedule,
                    rng_seed=run_seed,
                    robot_seed=robot_seed,
                    robot_radius=config_experiments.ROBOT_RADIUS
                )
                
                # Inject IDs and constants to create new columns in the DataFrame
                result["experiment_id"] = exp_idx
                result["simulation_id"] = sim_idx
                result["num_experiments"] = NUM_EXPERIMENTS
                result["num_simulations"] = NUM_SIMULATION
                
                print(f't_targets: {result["t_targets"]}, t_coverage: {result["t_coverage"]}, percent_coverage: {result["percent_coverage"]:.2f}%, mean_found: {result["mean_found"]:.4f}')
                rows.append(result)
    
    df = pd.DataFrame(rows)
    
    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures"], as_index=False)
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
        # Reorder columns slightly to make it readable in Excel
        cols = df.columns.tolist()
        cols = ['grid_size', 'n_robots', 'n_targets', 'n_failures', 'num_experiments', 'num_simulations', 'experiment_id', 'simulation_id'] + [c for c in cols if c not in ['grid_size', 'n_robots', 'n_targets', 'n_failures', 'num_experiments', 'num_simulations', 'experiment_id', 'simulation_id']]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization instead of headless experiments"
    )
    args = parser.parse_args()

    if args.visualize:
        visualization_script = CURRENT_DIR / "stigmergy_random_search.py"
        subprocess.run([sys.executable, str(visualization_script)])
    else:
        run_experiments()
