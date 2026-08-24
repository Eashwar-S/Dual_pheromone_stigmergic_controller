"""Analyze and visualize rendezvous convergence experiment results.

The script intentionally produces only two figures:
1. target-finding cost by distance shell and number of search robots;
2. average swarm convergence curves over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to create the two summary figures. "
        "Install the project requirements with `python -m pip install -r requirements.txt` "
        "or run this script from an environment that already has matplotlib."
    ) from exc


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = CURRENT_DIR / "experiment_results_rendezvous"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "analysis_outputs_rendezvous"
Z_95 = 1.96
RUN_KEYS = [
    "grid_size",
    "n_search_robots",
    "requested_distance_shell",
    "spawn_position_id",
    "algorithm_seed",
]
COLORS = {
    1: "#4c78a8",
    2: "#f58518",
    3: "#54a24b",
}


def mean_ci_summary(data: pd.DataFrame, groups: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["runs"] = len(group)

        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = Z_95 * sem
        rows.append(row)
    return pd.DataFrame(rows)


def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = results_dir / "results.xlsx"
    robot_metrics_path = results_dir / "search_robot_metrics.csv"
    swarm_path = results_dir / "swarm_convergence.csv"

    missing = [path for path in [workbook, robot_metrics_path, swarm_path] if not path.exists()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing rendezvous result files:\n{missing_list}")

    detailed = pd.read_excel(workbook, sheet_name="detailed")
    robot_metrics = pd.read_csv(robot_metrics_path)
    swarm = pd.read_csv(swarm_path)

    detailed["success_rate"] = detailed["found"].astype(float)
    detailed["steps_per_start_distance"] = detailed["steps_to_target"] / detailed[
        "actual_min_start_distance"
    ].replace(0, np.nan)
    detailed["detection_fraction_of_total_time"] = detailed["time_to_attractive_detection"] / detailed[
        "steps_to_target"
    ].replace(0, np.nan)
    return detailed, robot_metrics, swarm


def prepare_output(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for old_figure in figure_dir.glob("*.png"):
        old_figure.unlink()
    return figure_dir


def plot_target_cost(summary: pd.DataFrame, detailed: pd.DataFrame, figure_dir: Path) -> Path:
    grid_sizes = sorted(detailed["grid_size"].unique())
    shells = sorted(detailed["requested_distance_shell"].unique())
    robot_counts = sorted(detailed["n_search_robots"].unique())

    fig, axes = plt.subplots(2, len(grid_sizes), figsize=(6.0 * len(grid_sizes), 8.0), sharex=True)
    if len(grid_sizes) == 1:
        axes = np.array(axes).reshape(2, 1)

    width = min(0.78 / max(len(robot_counts), 1), 0.24)
    x = np.arange(len(shells), dtype=float)

    for col, grid_size in enumerate(grid_sizes):
        grid_summary = summary[summary["grid_size"] == grid_size]
        for idx, n_search_robots in enumerate(robot_counts):
            group = (
                grid_summary[grid_summary["n_search_robots"] == n_search_robots]
                .set_index("requested_distance_shell")
                .reindex(shells)
            )
            offset = (idx - (len(robot_counts) - 1) / 2) * width
            color = COLORS.get(int(n_search_robots))

            axes[0, col].bar(
                x + offset,
                group["steps_to_target_mean"],
                width,
                yerr=group["steps_to_target_ci95"],
                capsize=3,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{n_search_robots} searcher{'s' if n_search_robots != 1 else ''}",
            )
            axes[1, col].bar(
                x + offset,
                group["slowdown_ratio_mean"],
                width,
                yerr=group["slowdown_ratio_ci95"],
                capsize=3,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )

        axes[0, col].set_title(f"Grid {grid_size}x{grid_size}: steps to target")
        axes[1, col].set_title(f"Grid {grid_size}x{grid_size}: slowdown vs Manhattan")
        for row in range(2):
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(shells)
            axes[row, col].grid(axis="y", alpha=0.25)
            axes[row, col].set_xlabel("Requested start distance shell")

    axes[0, 0].set_ylabel("Mean steps to target")
    axes[1, 0].set_ylabel("Mean slowdown ratio")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Rendezvous Target-Finding Cost", y=0.995)
    fig.tight_layout()

    output_path = figure_dir / "01_target_finding_cost.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_convergence_curves(detailed: pd.DataFrame, swarm: pd.DataFrame, samples_per_run: int = 80) -> pd.DataFrame:
    initial = detailed[RUN_KEYS + ["actual_min_start_distance", "steps_to_target"]].copy()
    merged = swarm.merge(initial, on=RUN_KEYS, how="left", validate="many_to_one")
    if merged["actual_min_start_distance"].isna().any():
        raise ValueError("Some swarm convergence rows could not be matched to detailed run metadata.")

    curve_rows = []
    sample_points = np.linspace(0.0, 1.0, samples_per_run + 1)

    for keys, run in merged.groupby(RUN_KEYS, sort=False):
        run = run.sort_values("step")
        total_steps = float(run["steps_to_target"].iloc[0])
        if not np.isfinite(total_steps) or total_steps <= 0:
            continue

        normalized_step = (run["step"].to_numpy(dtype=float) / total_steps).clip(0.0, 1.0)
        normalized_distance = (
            run["min_distance_to_target_over_time"].to_numpy(dtype=float)
            / float(run["actual_min_start_distance"].iloc[0])
        )
        normalized_distance = np.clip(normalized_distance, 0.0, None)

        order = np.argsort(normalized_step, kind="stable")
        normalized_step = normalized_step[order]
        normalized_distance = normalized_distance[order]

        unique_step, unique_index = np.unique(normalized_step, return_index=True)
        unique_distance = normalized_distance[unique_index]
        interp_distance = np.interp(sample_points, unique_step, unique_distance)
        interp_distance[-1] = 0.0

        key_values = dict(zip(RUN_KEYS, keys))
        for progress, distance in zip(sample_points, interp_distance):
            curve_rows.append(
                {
                    **key_values,
                    "run_progress": progress,
                    "normalized_min_distance": distance,
                }
            )

    curves = pd.DataFrame(curve_rows)
    return mean_ci_summary(
        curves,
        ["grid_size", "n_search_robots", "run_progress"],
        ["normalized_min_distance"],
    )


def plot_convergence_curves(curves: pd.DataFrame, detailed: pd.DataFrame, figure_dir: Path) -> Path:
    grid_sizes = sorted(detailed["grid_size"].unique())
    robot_counts = sorted(detailed["n_search_robots"].unique())
    fig, axes = plt.subplots(1, len(grid_sizes), figsize=(6.0 * len(grid_sizes), 4.8), sharey=True)
    if len(grid_sizes) == 1:
        axes = np.array([axes])

    for ax, grid_size in zip(axes, grid_sizes):
        subset = curves[curves["grid_size"] == grid_size]
        for n_search_robots in robot_counts:
            line = subset[subset["n_search_robots"] == n_search_robots].sort_values("run_progress")
            if line.empty:
                continue
            x = line["run_progress"].to_numpy(dtype=float)
            mean = line["normalized_min_distance_mean"].to_numpy(dtype=float)
            ci = line["normalized_min_distance_ci95"].fillna(0.0).to_numpy(dtype=float)
            color = COLORS.get(int(n_search_robots))
            label = f"{n_search_robots} searcher{'s' if n_search_robots != 1 else ''}"
            ax.plot(x, mean, color=color, linewidth=2.0, label=label)
            ax.fill_between(x, np.maximum(mean - ci, 0.0), mean + ci, color=color, alpha=0.14, linewidth=0)

        ax.set_title(f"Grid {grid_size}x{grid_size}")
        ax.set_xlabel("Fraction of each run's target-finding time")
        ax.grid(alpha=0.25)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("Min distance to target / initial min distance")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Average Swarm Convergence Trajectory", y=0.995)
    fig.tight_layout()

    output_path = figure_dir / "02_swarm_convergence_trajectory.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_summaries(
    detailed: pd.DataFrame,
    robot_metrics: pd.DataFrame,
    performance_summary: pd.DataFrame,
    convergence_curves: pd.DataFrame,
    output_dir: Path,
) -> None:
    robot_summary = mean_ci_summary(
        robot_metrics,
        ["grid_size", "n_search_robots", "requested_distance_shell"],
        [
            "found",
            "steps_to_target",
            "time_to_attractive_detection",
            "time_from_detection_to_target",
            "path_efficiency",
            "post_detection_path_efficiency",
            "convergence_rate",
            "stagnation_count",
            "post_detection_stagnation_count",
            "final_distance",
        ],
    )
    overall = mean_ci_summary(
        detailed,
        ["grid_size", "n_search_robots"],
        [
            "success_rate",
            "steps_to_target",
            "slowdown_ratio",
            "path_efficiency",
            "time_to_attractive_detection",
            "time_from_detection_to_target",
            "convergence_rate",
            "detection_fraction_of_total_time",
        ],
    )

    detailed.to_csv(output_dir / "rendezvous_detailed_with_derived_metrics.csv", index=False)
    performance_summary.to_csv(output_dir / "rendezvous_performance_by_shell.csv", index=False)
    robot_summary.to_csv(output_dir / "rendezvous_robot_summary_by_shell.csv", index=False)
    overall.to_csv(output_dir / "rendezvous_overall_summary.csv", index=False)
    convergence_curves.to_csv(output_dir / "rendezvous_convergence_curve_summary.csv", index=False)


def print_audit(detailed: pd.DataFrame, robot_metrics: pd.DataFrame, swarm: pd.DataFrame) -> None:
    print("\nDATA AUDIT")
    print(f"Run-level rows: {len(detailed):,}")
    print(f"Robot-level rows: {len(robot_metrics):,}")
    print(f"Swarm time-series rows: {len(swarm):,}")
    counts = (
        detailed.groupby(["grid_size", "n_search_robots", "requested_distance_shell"])
        .size()
        .rename("runs")
        .reset_index()
    )
    print("\nRuns per scenario")
    print(counts.to_string(index=False))


def print_key_insights(detailed: pd.DataFrame, performance_summary: pd.DataFrame) -> None:
    print("\nKEY INSIGHTS")
    success_rate = detailed["found"].mean()
    print(f"- Overall success rate: {success_rate:.1%} across {len(detailed):,} runs.")

    best_steps = performance_summary.loc[performance_summary["steps_to_target_mean"].idxmin()]
    best_efficiency = performance_summary.loc[performance_summary["slowdown_ratio_mean"].idxmin()]
    worst_cost = performance_summary.loc[performance_summary["slowdown_ratio_mean"].idxmax()]
    print(
        "- Fastest scenario: "
        f"grid={int(best_steps['grid_size'])}, searchers={int(best_steps['n_search_robots'])}, "
        f"shell={int(best_steps['requested_distance_shell'])}, "
        f"mean steps={best_steps['steps_to_target_mean']:.1f}."
    )
    print(
        "- Most efficient scenario vs shortest Manhattan path: "
        f"grid={int(best_efficiency['grid_size'])}, searchers={int(best_efficiency['n_search_robots'])}, "
        f"shell={int(best_efficiency['requested_distance_shell'])}, "
        f"mean slowdown={best_efficiency['slowdown_ratio_mean']:.2f}x."
    )
    print(
        "- Highest-cost scenario: "
        f"grid={int(worst_cost['grid_size'])}, searchers={int(worst_cost['n_search_robots'])}, "
        f"shell={int(worst_cost['requested_distance_shell'])}, "
        f"mean slowdown={worst_cost['slowdown_ratio_mean']:.2f}x."
    )

    robot_effect = (
        detailed.groupby("n_search_robots")["steps_to_target"]
        .mean()
        .sort_index()
        .rename("mean_steps")
    )
    print("\nMean steps by number of search robots")
    print(robot_effect.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_dir = prepare_output(output_dir)

    detailed, robot_metrics, swarm = load_results(results_dir)
    performance_summary = mean_ci_summary(
        detailed,
        ["grid_size", "n_search_robots", "requested_distance_shell"],
        [
            "success_rate",
            "steps_to_target",
            "slowdown_ratio",
            "path_efficiency",
            "time_to_attractive_detection",
            "time_from_detection_to_target",
            "convergence_rate",
            "stagnation_count",
            "detection_fraction_of_total_time",
        ],
    )
    convergence_curves = build_convergence_curves(detailed, swarm)

    figure_paths = [
        plot_target_cost(performance_summary, detailed, figure_dir),
        plot_convergence_curves(convergence_curves, detailed, figure_dir),
    ]
    write_summaries(detailed, robot_metrics, performance_summary, convergence_curves, output_dir)

    print_audit(detailed, robot_metrics, swarm)
    print_key_insights(detailed, performance_summary)
    print(f"\nSaved CSV summaries to: {output_dir}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
