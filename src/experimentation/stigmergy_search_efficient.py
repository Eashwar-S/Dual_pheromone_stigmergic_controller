from pathlib import Path
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.communication_tracker import CommunicationTracker
from src.stigmergy.pheromone import apply_decay
from src.stigmergy.robot_efficient import Robot
from src.experimentation import config


def run_simulation(grid_size: int, n_robots: int, n_targets: int,
                   failure_schedule: List[Tuple[int, int]], rng_seed: int,
                   robot_radius: int = 5) -> Dict[str, object]:
    """Run headless stigmergy search simulation and return comprehensive metrics."""
    rng = np.random.default_rng(rng_seed)
    W = H = grid_size
    max_horizon = config.calculate_horizon(grid_size, n_robots)
    
    # Grid-adaptive deposit: calibrated so tanh(pher_sum/sigma) ≈ tanh(2) ≈ 0.96 at fresh cells.
    # sigma = 0.2*W, footprint K=25 cells, target ratio = 2 → D = 2*(0.2*W)/25 = 0.016*W
    # Ratio=5 was too strong (territory-locking), ratio=2 gives strong avoidance without traps.
    PHER_DEPOSIT = 1.0#0.016 * grid_size
    # Grid-adaptive decay: pheromone lasts ~5% of one robot's fair-share coverage time.
    # Ensures failed-robot zones become attractive again within the failure window (steps 150-400).
    TAU_DECAY = max(100.0, (grid_size ** 2 / n_robots) * 0.05)
    PHER_MIN = 1e-6
    COLLISION_RADIUS = 0  # 1 = 3x3 block safe
    # Longer escape bursts so robots clear large pheromone zones (doubled from //20)
    ESCAPE_LEN = max(10, grid_size // 10)
    # Trigger escapes sooner: max(3, grid_size//50) fires at 3-10 steps (was 5-16)
    STAGNANT_THRESH = max(3, grid_size // 50)
    
    # Generate random starting positions for robots
    start_positions = config.generate_robot_positions(grid_size, n_robots, rng)
    
    robots = [Robot(
        i,
        start_positions[i][0],
        start_positions[i][1],
        local_covered=np.zeros((H, W), dtype=bool),
        # escape_len=ESCAPE_LEN,
        # stagnant_thresh=STAGNANT_THRESH,
    ) for i in range(n_robots)]
    
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Tuple[int, int]] = set()
    covered = np.zeros((H, W), dtype=int)
    pheromone = np.zeros((H, W), dtype=float)
    
    # Process failure schedule
    norm_sched: List[Tuple[int, int]] = []
    for rid, st in (failure_schedule or []):
        if 0 <= rid < n_robots and st is not None and st >= 0:
            norm_sched.append((int(rid), int(st)))
    norm_sched.sort(key=lambda x: (x[1], x[0]))
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)
    
    t_targets = max_horizon
    t_coverage = max_horizon
    total_cells = H * W
    
    # Initialize communication tracker (stigmergy has no communication)
    comm_tracker = CommunicationTracker(message_size_bytes=5)
    
    auc_cov = 0.0
    auc_found = 0.0
    
    steps = 0
    while steps < max_horizon:
        # Apply pheromone decay
        apply_decay(pheromone, TAU_DECAY, PHER_MIN)
        
        # Handle failures
        if steps in fail_map:
            for rid in fail_map[steps]:
                r = robots[rid]
                if not r.failed:
                    r.failed = True
        
        # Pre-move sensing
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
        
        # Execute moves
        for r in robots:
            if not r.failed:
                r.step(pheromone, robots, robot_radius, COLLISION_RADIUS)
                # Deposit pheromone after moving
                r.deposit_pheromone(pheromone, PHER_DEPOSIT, robot_radius)
        
        # Post-move sensing
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
        
        # Track communication: stigmergy has no communication
        comm_tracker.record_step(0)
        
        steps += 1
        
        # Calculate metrics
        current_coverage = np.sum(covered > 0)
        coverage_fraction = current_coverage / total_cells
        targets_fraction = len(found_targets) / len(targets) if len(targets) > 0 else 1.0
        
        auc_cov += coverage_fraction
        auc_found += targets_fraction
        
        # Track completion times
        if len(found_targets) >= len(targets) and t_targets == max_horizon:
            t_targets = steps
        
        if current_coverage >= total_cells and t_coverage == max_horizon:
            t_coverage = steps
        
        # Early termination if both objectives met
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
    
    # Get communication metrics (will be all zeros for stigmergy)
    comm_metrics = comm_tracker.get_metrics()
    
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
        "TAU_DECAY": round(TAU_DECAY, 1),
        "BIAS_ALPHA": 1.0,  # Robot alpha
        "sigma": 0.2 * grid_size,
        "escape_len": ESCAPE_LEN,
        "beta": 0.3,        # Robot beta
        "_stagnant_count": STAGNANT_THRESH,
        # "t_end": t_end,
        # "success_targets": success_targets,
        # "success_coverage": success_coverage,
        # "success_both": success_both,
        # "auc_cov": auc_cov,
        # "mean_cov": mean_cov,
        # "auc_found": auc_found,
        # "pct_revisited": pct_revisited,
        # # Communication bandwidth metrics (all zeros for stigmergy)
        # "total_messages": comm_metrics['total_messages'],
        # "total_bandwidth_bytes": comm_metrics['total_bandwidth_bytes'],
        # "peak_messages_per_step": comm_metrics['peak_messages_per_step'],
        # "peak_bandwidth_bytes": comm_metrics['peak_bandwidth_bytes'],
        # "avg_messages_per_step": comm_metrics['avg_messages_per_step'],
    }


def run_experiments():
    """Run all experiments and save results to Excel."""
    output_path = config.get_output_path("stigmergy_search_efficient")
    
    rows: List[Dict[str, object]] = []
    configs_list = config.get_experiment_configs()

    for grid_size, n_robots, n_targets, n_failures in configs_list:
        if n_failures > 0:
            xlsx_path = output_path / "E2/results.xlsx"
        else:
            xlsx_path = output_path / "E1/results.xlsx"
        break
    
    for grid_size, n_robots, n_targets, n_failures in configs_list:
        max_horizon = config.calculate_horizon(grid_size, n_robots)
        
        schedule_seed = 42#(config.BASE_SEED * 10_000) + (grid_size * 100) + (n_robots * 10) + (n_failures * 1000)
        rng_sched = np.random.default_rng(schedule_seed)
        failure_schedule = config.make_random_failure_schedule(n_robots, n_failures, rng_sched, max_horizon)
        
        for run_idx in range(1, config.RUNS_PER_SCENARIO + 1):
            run_seed = schedule_seed#schedule_seed + run_idx
            
            print(f"Running: grid={grid_size}, robots={n_robots}, targets={n_targets}, failures={n_failures}, run={run_idx}")
            
            result = run_simulation(
                grid_size=grid_size,
                n_robots=n_robots,
                n_targets=n_targets,
                failure_schedule=failure_schedule,
                rng_seed=run_seed,
                robot_radius=config.ROBOT_RADIUS
            )
            print(f"t_targets: {result['t_targets']}, t_coverage: {result['t_coverage']}, percent_coverage: {result['percent_coverage']}, mean_found: {result['mean_found']}")
            
            rows.append(result)
    
    df = pd.DataFrame(rows)
    
    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures", "TAU_DECAY", "BIAS_ALPHA", "sigma", "escape_len", "beta", "_stagnant_count"], as_index=False)
          .agg(
              runs=("t_targets", "size"),
              avg_n_targets_found=("n_targets_found", "mean"),
              avg_t_targets=("t_targets", "mean"),
              avg_t_coverage=("t_coverage", "mean"),
              avg_percent_coverage=("percent_coverage", "mean"),
              avg_mean_found=("mean_found", "mean"),
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
