from pathlib import Path
from typing import List, Tuple, Set, Dict
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.utilities import generate_unique_targets, mark_visible, discover_targets_in_vnhood
from src.common.communication_tracker import CommunicationTracker
from src.stigmergy.pheromone import deposit_uniform, apply_decay
from src.stigmergy.robot_random import Robot
from src.experimentation import config


def run_simulation(grid_size: int, n_robots: int, n_targets: int,
                   failure_schedule: List[Tuple[int, int]], rng_seed: int,
                   robot_radius: int = 5) -> Dict[str, object]:
    """Run headless stigmergy random walk simulation and return comprehensive metrics."""
    rng = np.random.default_rng(rng_seed)
    W = H = grid_size
    max_horizon = config.calculate_horizon(grid_size, n_robots)

    PHER_DEPOSIT = 1.0
    TAU_DECAY = 200.0
    PHER_MIN = 1e-6
    BIAS_ALPHA = 0.5
    UNCOVERED_BONUS = 2.0
    COLLISION_RADIUS = 0  # 1 = 3x3 block safe zone

    # Generate random starting positions for robots
    start_positions = config.generate_robot_positions(grid_size, n_robots, rng)

    robots = [Robot(
        i,
        start_positions[i][0],
        start_positions[i][1],
        local_covered=np.zeros((H, W), dtype=bool)
    ) for i in range(n_robots)]

    targets = generate_unique_targets(grid_size, n_targets, rng)
    found_targets: Set[Tuple[int, int]] = set()
    covered = np.zeros((H, W), dtype=int)
    pheromone = np.zeros((H, W), dtype=float)

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
        apply_decay(pheromone, TAU_DECAY, PHER_MIN)

        if steps in fail_map:
            for rid in fail_map[steps]:
                r = robots[rid]
                if not r.failed:
                    r.failed = True

        # Sense before move
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)

        # Move + deposit
        for r in robots:
            if not r.failed:
                r.step(pheromone, BIAS_ALPHA, UNCOVERED_BONUS, rng, robots, COLLISION_RADIUS)
                deposit_uniform(pheromone, r.x, r.y, PHER_DEPOSIT, r=robot_radius)

        # Sense after move
        for r in robots:
            if not r.failed:
                mark_visible(r.local_covered, r.x, r.y, robot_radius)
                mark_visible(covered, r.x, r.y, robot_radius)
                discover_targets_in_vnhood(r.x, r.y, targets, found_targets, W, H, robot_radius)

        # Track communication: stigmergy has no communication
        comm_tracker.record_step(0)

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

    current_coverage = np.sum(covered > 0)
    percent_coverage = (current_coverage / total_cells) * 100.0

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
        "TAU_DECAY": TAU_DECAY,
        "BIAS_ALPHA": BIAS_ALPHA,
    }


def _run_one(args: Dict[str, object]) -> Dict[str, object]:
    """Pickle-safe worker wrapper."""
    return run_simulation(
        grid_size=int(args["grid_size"]),
        n_robots=int(args["n_robots"]),
        n_targets=int(args["n_targets"]),
        failure_schedule=args["failure_schedule"],
        rng_seed=int(args["run_seed"]),
        robot_radius=int(args["robot_radius"]),
    )


def run_experiments():
    """Run all experiments and save results to Excel (parallelized)."""
    output_path = config.get_output_path("stigmergy_random")
    configs_list = config.get_experiment_configs()

    # Decide output file (same logic you had)
    xlsx_path = None
    for grid_size, n_robots, n_targets, n_failures in configs_list:
        if n_failures > 0:
            xlsx_path = output_path / "E2/results.xlsx"
        else:
            xlsx_path = output_path / "E1/results.xlsx"
        break
    if xlsx_path is None:
        raise RuntimeError("No experiment configs returned by config.get_experiment_configs().")

    # How many workers?
    # Leave 1 core free by default; override via env var if you want.
    cpu = os.cpu_count() or 1
    default_workers = max(1, cpu - 1)
    max_workers = int(os.environ.get("N_WORKERS", default_workers))

    rows: List[Dict[str, object]] = []

    for grid_size, n_robots, n_targets, n_failures in configs_list:
        max_horizon = config.calculate_horizon(grid_size, n_robots)

        # Keep schedule fixed per scenario (good), but vary run_seed per run (important!)
        schedule_seed = 42
        rng_sched = np.random.default_rng(schedule_seed)
        failure_schedule = config.make_random_failure_schedule(
            n_robots, n_failures, rng_sched, max_horizon
        )

        jobs: List[Dict[str, object]] = []
        for run_idx in range(1, config.RUNS_PER_SCENARIO + 1):
            run_seed = schedule_seed #+ run_idx  # FIX: was constant before
            jobs.append({
                "grid_size": grid_size,
                "n_robots": n_robots,
                "n_targets": n_targets,
                "failure_schedule": failure_schedule,
                "run_seed": run_seed,
                "robot_radius": config.ROBOT_RADIUS,
            })

        print(f"\nScenario: grid={grid_size}, robots={n_robots}, targets={n_targets}, failures={n_failures}")
        print(f"Dispatching {len(jobs)} runs with max_workers={max_workers}")

        # Parallelize runs for this scenario
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run_one, job): job for job in jobs}
            completed = 0
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    res = fut.result()
                    rows.append(res)
                except Exception as e:
                    print(f"FAILED run_seed={job['run_seed']} scenario=(g={grid_size}, r={n_robots}, f={n_failures}) err={e}")
                    # Optionally re-raise:
                    # raise
                completed += 1
                if completed % max(1, len(jobs) // 10) == 0 or completed == len(jobs):
                    print(f"  progress: {completed}/{len(jobs)}")

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["grid_size", "n_robots", "n_failures", "TAU_DECAY", "BIAS_ALPHA"], as_index=False)
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
    # On Windows this guard is REQUIRED for multiprocessing.
    run_experiments()