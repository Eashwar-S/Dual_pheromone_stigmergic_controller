from pathlib import Path
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
import sys
import argparse
import subprocess
import random
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
import config_experiments
from environment import MultiAgentGridEnv
from model import DQNPolicy

def run_simulation(grid_size: int, n_robots: int, n_targets: int,
                   failure_schedule: List[Tuple[int, int]], rng_seed: int,
                   robot_seed: int, robot_radius: int, policy_net, device) -> Dict[str, object]:
    
    rng = np.random.default_rng(rng_seed)
    random.seed(robot_seed)
    torch.manual_seed(robot_seed)
    
    W = H = grid_size
    max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
    
    env = MultiAgentGridEnv(grid_size=grid_size, num_agents=n_robots, num_targets=n_targets, num_obstacles=0)
    env.reset()
    
    start_positions = config_experiments.generate_robot_positions(grid_size, n_robots, rng)
    env.agents = {i: start_positions[i] for i in range(n_robots)}
    
    initial_targets = generate_unique_targets(grid_size, n_targets, rng)
    env.targets = set(initial_targets)
    env.obstacles = set()
    env.visited_memory = {i: np.zeros((grid_size, grid_size)) for i in range(n_robots)}
    for i, pos in env.agents.items():
        env.visited_memory[i][pos] = 1
        
    obs_dict = env._get_all_observations()

    found_targets: Set[Tuple[int, int]] = set()
    covered_global = np.zeros((H, W), dtype=bool)
    
    norm_sched = [(int(rid), int(st)) for rid, st in (failure_schedule or []) if 0 <= rid < n_robots and st is not None and st >= 0]
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)
        
    t_targets, t_coverage = max_horizon, max_horizon
    total_cells = H * W
    auc_cov, auc_found = 0.0, 0.0
    steps = 0
    failed_robots = set()
    
    while steps < max_horizon:
        if steps in fail_map:
            for rid in fail_map[steps]:
                failed_robots.add(rid)

        action_dict = {}
        for agent_id, obs in obs_dict.items():
            if agent_id not in failed_robots:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(obs_tensor)
                action_dict[agent_id] = q_values.argmax().item()
                
        if len(failed_robots) == n_robots:
            break

        if len(action_dict) > 0:
            next_obs_dict, _, done, _ = env.step(action_dict)
            obs_dict = next_obs_dict
        else:
            done = (len(env.targets) == 0)

        for agent_id, pos in env.agents.items():
            if agent_id not in failed_robots:
                mark_visible(covered_global, pos[0], pos[1], robot_radius)
                discover_targets_in_vnhood(pos[0], pos[1], initial_targets, found_targets, W, H, robot_radius)
                
        steps += 1
        current_coverage = np.sum(covered_global)
        auc_cov += current_coverage / total_cells
        auc_found += len(found_targets) / len(initial_targets) if len(initial_targets) > 0 else 1.0
        
        if len(found_targets) >= len(initial_targets) and t_targets == max_horizon:
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
        "n_targets": len(initial_targets),
        "n_failures": len(norm_sched),
        "n_targets_found": len(found_targets),
        "t_targets": t_targets,
        "t_coverage": t_coverage,
        "percent_coverage": percent_coverage,
        "mean_found": mean_found,
    }

def run_experiments():
    output_path = Path.cwd() / "experiment_results_marl"
    output_path.mkdir(exist_ok=True)
    NUM_EXPERIMENTS = 10
    NUM_SIMULATIONS = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQNPolicy(input_channels=4, fov_size=3, action_space=4).to(device)
    checkpoint_path = Path("checkpoints/best_policy.pth")
    if checkpoint_path.exists():
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        policy_net.eval()
        print(f"Loaded best policy from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")
    
    rows: List[Dict[str, object]] = []
    configs_list = config_experiments.get_experiment_configs()
    is_failure_experiment = any(cfg[3] > 0 for cfg in configs_list)
    failure_time_modes = list(config_experiments.FAILURE_TIME_WINDOWS) if is_failure_experiment else [config_experiments.FAILURE_TIME_MODE]

    if is_failure_experiment:
        xlsx_path = output_path / "E2/results.xlsx"
        dir_path = output_path / "E2"
    else:
        xlsx_path = output_path / "E1/results.xlsx"
        dir_path = output_path / "E1"
    dir_path.mkdir(parents=True, exist_ok=True)

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
                for sim_idx in range(1, NUM_SIMULATIONS + 1):
                    robot_seed = run_seed * 1000 + sim_idx 
                    
                    print(f"Running MARL Experiment: grid={grid_size}, robots={n_robots}, targets={n_targets}, failures={n_failures}, failure_time_mode={failure_time_mode}, exp={exp_idx}, sim={sim_idx}")
                    
                    result = run_simulation(
                        grid_size=grid_size,
                        n_robots=n_robots,
                        n_targets=n_targets,
                        failure_schedule=failure_schedule,
                        rng_seed=run_seed,
                        robot_seed=robot_seed,
                        robot_radius=config_experiments.ROBOT_RADIUS,
                        policy_net=policy_net,
                        device=device
                    )
                    
                    result["failure_time_mode"] = failure_time_mode
                    result["experiment_id"] = exp_idx
                    result["simulation_id"] = sim_idx
                    result["num_experiments"] = NUM_EXPERIMENTS
                    result["num_simulations"] = NUM_SIMULATIONS
                    
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
        cols = df.columns.tolist()
        lead_cols = ['grid_size', 'n_robots', 'n_targets', 'n_failures', 'failure_time_mode', 'num_experiments', 'num_simulations', 'experiment_id', 'simulation_id']
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualization instead of headless experiments"
    )
    args = parser.parse_args()

    if args.visualize:
        visualization_script = CURRENT_DIR / "visualization.py"
        subprocess.run([sys.executable, str(visualization_script)])
    else:
        run_experiments()
