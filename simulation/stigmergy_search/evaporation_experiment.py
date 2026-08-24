import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import concurrent.futures
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from stigmergy_common.pheromone import apply_decay
from robot_efficient import Robot

def run_headless_sim(W, H, n_robots, n_targets, robot_radius, collision_radius, tau_decay, seed):
    rng = np.random.default_rng(seed)
    robots = [Robot(i, rng.integers(0, W), rng.integers(0, H), np.zeros((H, W), dtype=bool)) for i in range(n_robots)]
    covered = np.zeros((H, W), dtype=bool)
    targets = generate_unique_targets(W, n_targets, rng)
    found_targets = set()
    pheromone = np.zeros((H, W), dtype=float)
    
    targets_over_time = []
    global_step = 0
    
    while len(found_targets) < n_targets:
        apply_decay(pheromone, tau_decay, 1e-6)
        
        for r in robots:
            mark_visible(r.local_covered, r.x, r.y, robot_radius)
            mark_visible(covered, r.x, r.y, robot_radius)
            discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
            r.deposit_pheromone(pheromone, 1.0, robot_radius)
            
        for r in robots:
            r.step(pheromone, robots, robot_radius, collision_radius)
            mark_visible(r.local_covered, r.x, r.y, robot_radius)
            mark_visible(covered, r.x, r.y, robot_radius)
            discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
            
        global_step += 1
        targets_over_time.append(len(found_targets))
        
            
    return global_step, targets_over_time

def _worker_task(args):
    g_size, n_robots, n_targets, robot_radius, collision_radius, tau, seed, mult, run_idx = args
    steps, tot = run_headless_sim(g_size, g_size, n_robots, n_targets, robot_radius, collision_radius, tau, seed)
    return {
        'Grid_Size': g_size,
        'Tau_Multiplier': mult,
        'Run': run_idx,
        'Completion_Steps': steps,
        'Targets_Over_Time': str(tot)
    }

if __name__ == "__main__":
    N_ROBOTS_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    ROBOT_RADIUS = 1
    COLLISION_RADIUS = 1
    REPEATS = 10
    
    GRID_SIZES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    TAU_MULTIPLIERS = [0.5, 1.0, 2.0, 3.0, 4.0]
    COLORS = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#0571b0']
    
    # 1. Pre-compute all tasks to distribute to workers
    tasks = []
    for i, g_size in enumerate(GRID_SIZES):
        n_robots = N_ROBOTS_LIST[i]
        n_targets = 2 * n_robots
        base_tau = (g_size ** 2) / (n_robots * max(1, ROBOT_RADIUS))
        
        for run_idx in range(REPEATS):
            seed = int(g_size * 1000 + run_idx)
            for mult in TAU_MULTIPLIERS:
                tau = base_tau * mult
                tasks.append((g_size, n_robots, n_targets, ROBOT_RADIUS, COLLISION_RADIUS, tau, seed, mult, run_idx))

    results = []
    
    # 2. Execute pool. Max_workers=None defaults to the number of processors on the machine.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_worker_task, task) for task in tasks]
        
        with tqdm(total=len(tasks), desc="Simulating Swarm Scenarios (Parallel)") as pbar:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    # 3. Save & Plot
    df = pd.DataFrame(results)
    df.to_excel("stigmergy_experiment_results.xlsx", index=False)
    
    summary_df = df.groupby(['Grid_Size', 'Tau_Multiplier'])['Completion_Steps'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(10, 6))
    for idx, mult in enumerate(TAU_MULTIPLIERS):
        sub_df = summary_df[summary_df['Tau_Multiplier'] == mult]
        plt.errorbar(sub_df['Grid_Size'], sub_df['mean'], yerr=sub_df['std'], 
                     fmt='o:', color=COLORS[idx], capsize=4, label=f'Tau = {mult}x Base')
        
    plt.xlabel('Grid Size')
    plt.ylabel('Completion Steps (Mean)')
    plt.title('Time to Discover All Targets vs Grid Size (Controlled Layouts)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("tau_decay_experiment.png", dpi=300)
    plt.show()