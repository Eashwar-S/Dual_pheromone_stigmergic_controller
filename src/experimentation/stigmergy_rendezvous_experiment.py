from pathlib import Path
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.stigmergy.pheromone import deposit_uniform, apply_decay
from src.stigmergy.robot_rendezvous import Robot
from src.experimentation import config


def run_simulation(grid_size: int, n_robots: int, n_targets: int,
                   failure_schedule: List[Tuple[int, int]], rng_seed: int,
                   robot_radius: int = 5) -> Dict[str, object]:
    """Run headless stigmergy rendezvous (SEARCH-only) simulation and return comprehensive metrics."""
    rng = np.random.default_rng(rng_seed)
    W = H = grid_size
    max_horizon = config.calculate_horizon(grid_size, n_robots)
    
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 200.0
    PHER_MIN = 1e-6
    COLLISION_RADIUS = 1
    
    start_positions = []
    for i in range(n_robots):
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        start_positions.append((x, y))
    
    robots = [Robot(
        id=i,
        x=start_positions[i][0],
        y=start_positions[i][1],
        local_covered=np.zeros((H, W), dtype=bool),
        start_x=start_positions[i][0],
        start_y=start_positions[i][1]
    ) for i in range(n_robots)]
    
    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Tuple[int, int]] = set()
    global_target_visits: Dict[Tuple[int, int], Set[int]] = {}
    covered = np.zeros((H, W), dtype=int)
    pher_repulse = np.zeros((H, W), dtype=float)
    pher_attract = np.zeros((H, W), dtype=float)
    
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
    
    auc_cov = 0.0
    auc_found = 0.0
    
    steps = 0
    while steps < max_horizon:
        apply_decay(pher_repulse, TAU_DECAY, PHER_MIN)
        
        if steps in fail_map:
            for rid in fail_map[steps]:
                r = robots[rid]
                if not r.failed:
                    r.failed = True
        
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
        
        for r in robots:
            if not r.failed:
                if (r.x, r.y) in targets:
                    if (r.x, r.y) not in r.visited_targets:
                        r.visited_targets.add((r.x, r.y))
                        if (r.x, r.y) not in global_target_visits:
                            global_target_visits[(r.x, r.y)] = set()
                        global_target_visits[(r.x, r.y)].add(r.id)
                
                if r.mode != "ADVERTISE":
                    nearby_targets = []
                    R = robot_radius
                    for t in targets:
                        dist = abs(t[0] - r.x) + abs(t[1] - r.y)
                        if dist <= R:
                            nearby_targets.append((t, dist))
                    
                    if nearby_targets:
                        nearby_targets.sort(key=lambda x: x[1])
                        target_pos = nearby_targets[0][0]
                        dx = np.sign(target_pos[0] - r.x)
                        dy = np.sign(target_pos[1] - r.y)
                        if dx != 0 and dy != 0:
                            if rng.random() < 0.5:
                                dx = 0
                            else:
                                dy = 0
                        nx, ny = r.x + dx, r.y + dy
                        
                        if 0 <= nx < W and 0 <= ny < H and r.is_clear(nx, ny, robots, COLLISION_RADIUS):
                            r.x, r.y = nx, ny
                        continue
                
                if r.mode == "SEARCH":
                    found_trail = False
                    R = robot_radius
                    for dy in range(-R, R + 1):
                        for dx in range(-R, R + 1):
                            if abs(dx) + abs(dy) <= R:
                                cx, cy = r.x + dx, r.y + dy
                                if 0 <= cx < W and 0 <= cy < H:
                                    if pher_attract[cy, cx] > 1e-5:
                                        found_trail = True
                                        break
                        if found_trail:
                            break
                    
                    if found_trail:
                        r.mode = "FOLLOW"
                        r.follow_history.clear()
                        nx, ny = r.choose_move_follow(pher_attract, robots, COLLISION_RADIUS)
                        r.x, r.y = nx, ny
                    else:
                        nx, ny = r.choose_move_search(pher_repulse, pher_attract, robots, COLLISION_RADIUS)
                        r.x, r.y = nx, ny
                        deposit_uniform(pher_repulse, r.x, r.y, PHER_DEPOSIT, r=robot_radius)
                
                elif r.mode == "FOLLOW":
                    nx, ny = r.choose_move_follow(pher_attract, robots, COLLISION_RADIUS)
                    
                    if nx == r.x and ny == r.y:
                        if (nx, ny) not in targets:
                            r.mode = "SEARCH"
                    
                    r.x, r.y = nx, ny
        
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
        
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
    output_path = config.get_output_path("stigmergy_rendezvous")
    xlsx_path = output_path / "results.xlsx"
    
    rows: List[Dict[str, object]] = []
    configs_list = config.get_experiment_configs()
    
    for grid_size, n_robots, n_targets, n_failures in configs_list:
        max_horizon = config.calculate_horizon(grid_size, n_robots)
        
        schedule_seed = (config.BASE_SEED * 10_000) + (grid_size * 100) + (n_robots * 10) + (n_failures * 1000)
        rng_sched = np.random.default_rng(schedule_seed)
        failure_schedule = config.make_random_failure_schedule(n_robots, n_failures, rng_sched, max_horizon)
        
        for run_idx in range(1, config.RUNS_PER_SCENARIO + 1):
            run_seed = schedule_seed + run_idx
            
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
