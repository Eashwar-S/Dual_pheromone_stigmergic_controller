from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

import config_experiments
from common.experiment_outputs import autosize_excel_columns, split_scalar_and_timeseries, write_timeseries_csvs
from random_walk_experiment import run_simulation


NUM_EXPERIMENTS = 10
NUM_SIMULATIONS = 10
N_CURVE_SAMPLES = 201
FAILURE_TIME_MODES = ("mixed", "early", "middle", "late")


def _failure_time_modes(n_failures: int) -> Tuple[str, ...]:
    if n_failures > 0:
        return FAILURE_TIME_MODES
    return (config_experiments.FAILURE_TIME_MODE,)


def _run_task(task: Tuple[int, int, int, int, int, str, int, int, List[Tuple[int, int]], int, int]) -> Tuple[int, Dict[str, object]]:
    order, grid_size, n_robots, n_targets, n_failures, failure_time_mode, exp_idx, sim_idx, failure_schedule, run_seed, robot_seed = task

    print(
        f"Running Random Walk: grid={grid_size}, robots={n_robots}, targets={n_targets}, "
        f"failures={n_failures}, mode={failure_time_mode}, exp={exp_idx}, sim={sim_idx}",
        flush=True,
    )

    result = run_simulation(
        grid_size=grid_size,
        n_robots=n_robots,
        n_targets=n_targets,
        failure_schedule=failure_schedule,
        rng_seed=run_seed,
        robot_seed=robot_seed,
        robot_radius=config_experiments.ROBOT_RADIUS,
        collect_time_series=True,
        n_curve_samples=N_CURVE_SAMPLES,
    )

    result["experiment_id"] = exp_idx
    result["simulation_id"] = sim_idx
    result["failure_time_mode"] = failure_time_mode
    result["num_experiments"] = NUM_EXPERIMENTS
    result["num_simulations"] = NUM_SIMULATIONS
    return order, result


def _build_tasks() -> List[Tuple[int, int, int, int, int, str, int, int, List[Tuple[int, int]], int, int]]:
    tasks = []
    order = 0
    schedule_seed = 42

    for grid_size, n_robots, n_targets, n_failures in config_experiments.get_experiment_configs():
        max_horizon = config_experiments.calculate_horizon(grid_size, n_robots)
        for failure_time_mode in _failure_time_modes(n_failures):
            rng_sched = np.random.default_rng(schedule_seed)
            failure_schedule = config_experiments.make_random_failure_schedule(
                grid_size,
                n_robots,
                n_failures,
                rng_sched,
                max_horizon,
                failure_time_mode=failure_time_mode,
            )

            for exp_idx in range(1, NUM_EXPERIMENTS + 1):
                run_seed = schedule_seed + exp_idx
                for sim_idx in range(1, NUM_SIMULATIONS + 1):
                    robot_seed = run_seed * 1000 + sim_idx
                    tasks.append(
                        (
                            order,
                            grid_size,
                            n_robots,
                            n_targets,
                            n_failures,
                            failure_time_mode,
                            exp_idx,
                            sim_idx,
                            failure_schedule,
                            run_seed,
                            robot_seed,
                        )
                    )
                    order += 1

    return tasks


def _write_results(rows: List[Dict[str, object]]) -> Path:
    output_path = Path.cwd() / "experiment_results_stigmergy"
    output_path.mkdir(exist_ok=True)
    configs_list = config_experiments.get_experiment_configs()

    if configs_list and configs_list[0][3] > 0:
        xlsx_path = output_path / "E2/results.xlsx"
        dir_path = output_path / "E2"
    else:
        xlsx_path = output_path / "E1/results.xlsx"
        dir_path = output_path / "E1"
    dir_path.mkdir(exist_ok=True)

    scalar_rows, timeseries_rows = split_scalar_and_timeseries(rows, "random_walk")
    df = pd.DataFrame(scalar_rows)
    summary = (
        df.groupby(["algorithm", "grid_size", "n_robots", "n_failures", "failure_time_mode"], as_index=False)
        .agg(
            total_runs=("t_targets", "size"),
            avg_n_targets_found=("n_targets_found", "mean"),
            avg_t_targets=("t_targets", "mean"),
            avg_t_coverage=("t_coverage", "mean"),
            avg_percent_coverage=("percent_coverage", "mean"),
            avg_mean_found=("mean_found", "mean"),
            avg_coverage_auc=("coverage_auc_horizon_norm", "mean"),
            avg_target_auc=("target_auc_horizon_norm", "mean"),
            avg_redundancy=("avg_visits_per_covered_cell", "mean"),
            avg_extra_visits=("extra_visits_per_covered_cell", "mean"),
            avg_pct_revisited_cells=("pct_revisited_cells", "mean"),
            success_targets_rate=("success_targets", "mean"),
            success_coverage_rate=("success_coverage", "mean"),
        )
    )

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        cols = df.columns.tolist()
        lead_cols = [
            "grid_size",
            "n_robots",
            "n_targets",
            "n_failures",
            "failure_time_mode",
            "algorithm",
            "num_experiments",
            "num_simulations",
            "experiment_id",
            "simulation_id",
        ]
        df = df[lead_cols + [c for c in cols if c not in lead_cols]]

        df.to_excel(writer, index=False, sheet_name="detailed")
        summary.to_excel(writer, index=False, sheet_name="summary")

        autosize_excel_columns(writer, [("detailed", df), ("summary", summary)])

    write_timeseries_csvs(timeseries_rows, dir_path)

    return xlsx_path


def run_experiments(workers: Optional[int] = None) -> None:
    tasks = _build_tasks()
    max_workers = workers or os.cpu_count() or 1
    completed: Dict[int, Dict[str, object]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, task) for task in tasks]
        for future in as_completed(futures):
            order, result = future.result()
            completed[order] = result
            print(
                f"Completed {len(completed)}/{len(tasks)}: "
                f"t_targets={result['t_targets']}, t_coverage={result['t_coverage']}, "
                f"percent_coverage={result['percent_coverage']:.2f}%, mean_found={result['mean_found']:.4f}",
                flush=True,
            )

    rows = [completed[i] for i in range(len(tasks))]
    xlsx_path = _write_results(rows)
    print(f"\nResults saved to: {xlsx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes. Defaults to CPU count.")
    parser.add_argument("--visualize", action="store_true", help="Run visualization instead of headless experiments")
    args = parser.parse_args()

    if args.visualize:
        visualization_script = CURRENT_DIR / "random_search.py"
        subprocess.run([sys.executable, str(visualization_script)])
    else:
        run_experiments(workers=args.workers)
