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
                   robot_radius: int) -> Dict[str, object]:
    """Run headless stigmergy rendezvous simulation and return metrics aligned with stigmergy_search_efficient."""
    # Layout RNG: positions then target — must match centralized call order exactly
    rng_layout = np.random.default_rng(rng_seed)
    # Algorithm RNG: pheromone simulation execution (kept separate so layout is stable)
    rng_algo = np.random.default_rng(rng_seed + 1_000_000)
    W = H = grid_size
    max_horizon = config.calculate_horizon(grid_size, n_robots)

    PHER_DEPOSIT = 1.0
    TAU_DECAY = max(100.0, (grid_size ** 2 / n_robots) * 0.05)
    PHER_MIN = 1e-6
    COLLISION_RADIUS = 1

    # Layout: positions then target (same draw order as centralized for same seed)
    start_positions = config.generate_robot_positions(grid_size, n_robots, rng_layout)

    robots = [Robot(
        id=i,
        x=start_positions[i][0],
        y=start_positions[i][1],
        local_covered=np.zeros((H, W), dtype=bool),
        start_x=start_positions[i][0],
        start_y=start_positions[i][1],
    ) for i in range(n_robots)]

    targets = generate_unique_targets(grid_size, n_targets, rng_layout)
    found_targets: Set[Tuple[int, int]] = set()
    global_target_visits: Dict[Tuple[int, int], Set[int]] = {}
    covered = np.zeros((H, W), dtype=int)
    pher_repulse = np.zeros((H, W), dtype=float)
    pher_attract = np.zeros((H, W), dtype=float)

    # Process failure schedule
    norm_sched: List[Tuple[int, int]] = []
    for rid, st in (failure_schedule or []):
        if 0 <= rid < n_robots and st is not None and st >= 0:
            norm_sched.append((int(rid), int(st)))
    norm_sched.sort(key=lambda x: (x[1], x[0]))
    fail_map: Dict[int, List[int]] = {}
    for rid, st in norm_sched:
        fail_map.setdefault(st, []).append(rid)

    t_targets = max_horizon      # step when BOTH robots have detected the target
    t_first_detect = max_horizon # step when FIRST robot detects the target
    t_coverage = max_horizon
    total_cells = H * W
    robot_found: Set[int] = set()  # ids of robots that have detected the target

    auc_cov = 0.0
    auc_found = 0.0

    steps = 0
    while steps < max_horizon:
        apply_decay(pher_repulse, TAU_DECAY, PHER_MIN)
        apply_decay(pher_attract, TAU_DECAY, PHER_MIN)

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
                r.step(pher_repulse, pher_attract, targets, robots,
                       global_target_visits, COLLISION_RADIUS, PHER_DEPOSIT)

        # Post-move sensing + per-robot target detection
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)
                # Track which robots detect the single target
                if r.id not in robot_found:
                    tx, ty = next(iter(targets))
                    if abs(r.x - tx) + abs(r.y - ty) <= robot_radius:
                        robot_found.add(r.id)
                        if len(robot_found) == 1:
                            t_first_detect = steps
                        if len(robot_found) >= n_robots and t_targets == max_horizon:
                            t_targets = steps

        steps += 1

        current_coverage = np.sum(covered > 0)
        coverage_fraction = current_coverage / total_cells
        targets_fraction = len(found_targets) / len(targets) if len(targets) > 0 else 1.0

        auc_cov += coverage_fraction
        auc_found += targets_fraction

        if current_coverage >= total_cells and t_coverage == max_horizon:
            t_coverage = steps

        if t_targets < max_horizon and t_coverage < max_horizon:
            break

    t_end = steps
    mean_found = auc_found / t_end if t_end > 0 else 0.0

    current_coverage = np.sum(covered > 0)
    percent_coverage = (current_coverage / total_cells) * 100.0

    return {
        "grid_size": grid_size,
        "n_robots": n_robots,
        "n_targets": len(targets),
        "n_failures": len(norm_sched),
        "n_targets_found": len(found_targets),
        "t_first_detect": t_first_detect,
        "t_targets": t_targets,
        "t_coverage": t_coverage,
        "percent_coverage": percent_coverage,
        "mean_found": mean_found,
        "TAU_DECAY": round(TAU_DECAY, 1),
    }


def run_experiments():
    """Run all experiments and save results to Excel."""
    output_path = config.get_output_path("stigmergy_rendezvous")

    rows: List[Dict[str, object]] = []
    configs_list = config.get_experiment_configs()

    xlsx_path = output_path / "results.xlsx"
   
    # for grid_size, n_robots, n_targets, n_failures in configs_list:
    n_failures = 0
    n_targets = 1
    n_robots = 2
    grid_size = 100
    max_horizon = config.calculate_horizon(grid_size, n_robots)

    schedule_seed = 42
    rng_sched = np.random.default_rng(schedule_seed)
    failure_schedule = config.make_random_failure_schedule(n_robots, n_failures, rng_sched, max_horizon)

    for run_idx in range(1, 20):#config.RUNS_PER_SCENARIO + 1):
        # Unique seed per run so robot/target positions vary
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
        print(f"t_targets: {result['t_targets']}, t_coverage: {result['t_coverage']}, percent_coverage: {result['percent_coverage']}, mean_found: {result['mean_found']}")

        rows.append(result)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures", "TAU_DECAY"], as_index=False)
          .agg(
              runs=("t_targets", "size"),
              avg_n_targets_found=("n_targets_found", "mean"),
              avg_t_targets=("t_targets", "mean"),
              avg_t_coverage=("t_coverage", "mean"),
              avg_percent_coverage=("percent_coverage", "mean"),
              avg_mean_found=("mean_found", "mean"),
          )
    )

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
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
