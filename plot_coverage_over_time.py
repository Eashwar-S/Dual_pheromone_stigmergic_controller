# -*- coding: utf-8 -*-
"""Plot centralized, random-walk, and stigmergy RoboMaster coverage runs."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from plot_robot_coverage import (
    CoverageGrid,
    resolve_metadata_path,
    robot_position,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIRECTORY / "robomaster_results"
APPROACH_DIRECTORIES = {
    "centralized": "centralized_results",
    "random": "random_walk_results",
    "stigmergy": "stigmergy_results",
}
DEFAULT_EXPERIMENT = "E2"
DEFAULT_CELL_SIZE_METERS = 0.05
COMMON_COMPARISON_TIME_SECONDS = 200.0
ANALYSIS_THRESHOLDS = (50.0, 60.0, 70.0)


def natural_sort_key(path):
    """Sort run_2 before run_10 regardless of the run directory prefix."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(path))
    ]


def discover_experiment_runs(results_root, experiment):
    """Find each approach's metadata files in an E1 or E2 results branch."""
    results_root = Path(results_root).expanduser().resolve()
    discovered = {}
    for approach, directory_name in APPROACH_DIRECTORIES.items():
        experiment_directory = results_root / directory_name / experiment
        if not experiment_directory.is_dir():
            raise FileNotFoundError(
                "{0} results directory not found: {1}".format(
                    experiment, experiment_directory
                )
            )
        paths = sorted(
            experiment_directory.rglob("metadata.json"),
            key=natural_sort_key,
        )
        if not paths:
            raise FileNotFoundError(
                "No metadata.json files found below: {0}".format(
                    experiment_directory
                )
            )
        discovered[approach] = paths

    if len(discovered["centralized"]) != 1:
        raise ValueError(
            "Expected one centralized run for {0}, found {1}".format(
                experiment, len(discovered["centralized"])
            )
        )
    return discovered


def parse_timestamp(timestamp):
    """Convert an ISO UTC timestamp to a datetime."""
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    return datetime.fromisoformat(timestamp)


def load_metadata(path):
    path = resolve_metadata_path(path)
    with path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    samples = metadata.get("samples", [])
    if not samples:
        raise ValueError("No samples found in {0}".format(path))
    return path, samples


def elapsed_seconds(samples):
    start_time = parse_timestamp(samples[0]["timestamp_utc"])
    return np.array(
        [
            (parse_timestamp(sample["timestamp_utc"]) - start_time).total_seconds()
            for sample in samples
        ],
        dtype=float,
    )


def recorded_coverage_series(samples):
    """Read coverage.percent when it is recorded in every sample."""
    percentages = []
    for sample in samples:
        coverage = sample.get("coverage")
        if not isinstance(coverage, dict) or "percent" not in coverage:
            return None
        percentages.append(float(coverage["percent"]))
    return np.array(percentages, dtype=float)


def reconstructed_coverage_series(samples, cell_size):
    """Reconstruct cumulative coverage from swept robot footprints."""
    robot_names = sorted(
        {
            robot_name
            for sample in samples
            for robot_name in sample.get("robots", {})
        }
    )
    if not robot_names:
        raise ValueError("No robot positions are present in the metadata")

    grid = CoverageGrid(cell_size)
    previous_positions = {name: None for name in robot_names}
    percentages = []

    for sample in samples:
        for name in robot_names:
            current = robot_position(sample, name)
            if current is None:
                continue
            grid.add_swept_footprint(previous_positions[name], current)
            previous_positions[name] = current

        covered_cells = np.count_nonzero(grid.counts)
        percentages.append(100.0 * covered_cells / grid.counts.size)

    return np.array(percentages, dtype=float)


def load_coverage_run(path, cell_size):
    """Return label, elapsed seconds, and cumulative coverage percentage."""
    metadata_path, samples = load_metadata(path)
    percentages = recorded_coverage_series(samples)
    source = "recorded"
    if percentages is None:
        percentages = reconstructed_coverage_series(samples, cell_size)
        source = "reconstructed"

    return {
        "label": metadata_path.parent.name,
        "path": metadata_path,
        "time": elapsed_seconds(samples),
        "coverage": percentages,
        "source": source,
    }


def coverage_at_time(run, time_seconds):
    if time_seconds > run["time"][-1]:
        return float("nan")
    return float(np.interp(time_seconds, run["time"], run["coverage"]))


def time_to_coverage(run, threshold):
    indices = np.flatnonzero(run["coverage"] >= threshold)
    if not len(indices):
        return float("nan")
    return float(run["time"][indices[0]])


def normalized_auc(run):
    """Mean coverage over the run, expressed as a percentage."""
    duration = float(run["time"][-1])
    if duration <= 0.0:
        return float(run["coverage"][-1])
    return float(np.trapezoid(run["coverage"], run["time"]) / duration)


def build_analysis_rows(centralized, random_runs, stigmergy_runs):
    rows = []
    groups = (
        ("Centralized", [centralized]),
        ("Random walk", random_runs),
        ("Stigmergy", stigmergy_runs),
    )
    for method, runs in groups:
        for run in runs:
            row = {
                "method": method,
                "run": run["label"],
                "duration_seconds": float(run["time"][-1]),
                "final_coverage_percent": float(run["coverage"][-1]),
                "mean_coverage_auc_percent": normalized_auc(run),
                "coverage_at_200_seconds": coverage_at_time(
                    run, COMMON_COMPARISON_TIME_SECONDS
                ),
            }
            for threshold in ANALYSIS_THRESHOLDS:
                row["time_to_{0:.0f}_percent".format(threshold)] = (
                    time_to_coverage(run, threshold)
                )
            rows.append(row)
    return rows


def write_analysis_csv(rows, output_path):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def configure_axis(axis, title):
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Coverage (%)")
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, color="0.85", linewidth=0.8)


def plot_single_run(axis, run, color):
    axis.plot(
        run["time"],
        run["coverage"],
        color=color,
        linewidth=3.0,
        label="Centralized",
        zorder=5,
    )
    axis.scatter(
        run["time"][-1],
        run["coverage"][-1],
        color=color,
        s=35,
        zorder=6,
    )
    axis.annotate(
        "{0:.1f}%".format(run["coverage"][-1]),
        (run["time"][-1], run["coverage"][-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=color,
        fontsize=9,
    )
    axis.set_xlim(left=0.0)


def interpolate_runs(run_group, points=500):
    """Interpolate runs onto one common time axis for envelope shading."""
    common_duration = min(float(run["time"][-1]) for run in run_group)
    common_time = np.linspace(0.0, common_duration, points)
    interpolated = []

    for run in run_group:
        values = np.interp(
            common_time,
            run["time"],
            run["coverage"],
            left=run["coverage"][0],
        )
        interpolated.append(values)

    matrix = np.vstack(interpolated)
    lower = np.min(matrix, axis=0)
    upper = np.max(matrix, axis=0)
    mean = np.mean(matrix, axis=0)
    return common_time, lower, upper, mean


def plot_run_group(axis, runs, run_color, mean_color, method_name):
    for index, run in enumerate(runs):
        axis.plot(
            run["time"],
            run["coverage"],
            color=run_color,
            linewidth=1.15,
            alpha=0.42,
            label="{0} individual runs".format(method_name)
            if index == 0
            else None,
            zorder=2,
        )

    common_time, lower, upper, mean = interpolate_runs(runs)
    axis.fill_between(
        common_time,
        lower,
        upper,
        color=mean_color,
        alpha=0.14,
        label="{0} min-max range".format(method_name),
        zorder=1,
    )
    axis.plot(
        common_time,
        mean,
        color=mean_color,
        linewidth=3.0,
        linestyle="-",
        label="{0} mean".format(method_name),
        zorder=4,
    )
    return common_time, mean


def create_plot(
    centralized_path,
    random_paths,
    stigmergy_paths,
    output,
    csv_output,
    cell_size,
    show,
    experiment=None,
):
    centralized = load_coverage_run(centralized_path, cell_size)
    random_runs = [
        load_coverage_run(path, cell_size) for path in random_paths
    ]
    stigmergy_runs = [
        load_coverage_run(path, cell_size) for path in stigmergy_paths
    ]

    analysis_rows = build_analysis_rows(
        centralized, random_runs, stigmergy_runs
    )
    csv_path = write_analysis_csv(analysis_rows, csv_output)

    figure, axis = plt.subplots(figsize=(13.5, 8), constrained_layout=True)
    configure_axis(
        axis,
        "Coverage Over Time{0}: Centralized, Random Walk, and Stigmergy".format(
            " ({0})".format(experiment) if experiment else ""
        ),
    )
    plot_single_run(axis, centralized, "#1565c0")

    random_time, random_mean = plot_run_group(
        axis,
        random_runs,
        run_color="#8c9a8d",
        mean_color="#4f6d58",
        method_name="Random walk",
    )
    stigmergy_time, stigmergy_mean = plot_run_group(
        axis,
        stigmergy_runs,
        run_color="#f29b82",
        mean_color="#d84315",
        method_name="Stigmergy",
    )
    maximum_time = max(
        float(centralized["time"][-1]),
        *(float(run["time"][-1]) for run in random_runs),
        *(float(run["time"][-1]) for run in stigmergy_runs)
    )
    axis.set_xlim(0.0, maximum_time)
    axis.axvline(
        COMMON_COMPARISON_TIME_SECONDS,
        color="0.35",
        linestyle=":",
        linewidth=1.5,
        zorder=0,
    )

    random_at_comparison = np.array(
        [
            coverage_at_time(run, COMMON_COMPARISON_TIME_SECONDS)
            for run in random_runs
        ]
    )
    stigmergy_at_comparison = np.array(
        [
            coverage_at_time(run, COMMON_COMPARISON_TIME_SECONDS)
            for run in stigmergy_runs
        ]
    )
    random_average = float(np.nanmean(random_at_comparison))
    stigmergy_average = float(np.nanmean(stigmergy_at_comparison))
    random_std = float(np.nanstd(random_at_comparison, ddof=1))
    stigmergy_std = float(np.nanstd(stigmergy_at_comparison, ddof=1))
    advantage = stigmergy_average - random_average

    random_mean_at_comparison = float(
        np.interp(
            COMMON_COMPARISON_TIME_SECONDS, random_time, random_mean
        )
    )
    stigmergy_mean_at_comparison = float(
        np.interp(
            COMMON_COMPARISON_TIME_SECONDS,
            stigmergy_time,
            stigmergy_mean,
        )
    )
    axis.scatter(
        [COMMON_COMPARISON_TIME_SECONDS] * 2,
        [random_mean_at_comparison, stigmergy_mean_at_comparison],
        color=("#4f6d58", "#d84315"),
        s=55,
        zorder=6,
    )
    axis.annotate(
        "At 200 s, stigmergy leads by {0:+.1f} percentage points\n"
        "Mean ± SD: {1:.1f} ± {2:.1f}% vs {3:.1f} ± {4:.1f}%".format(
            advantage,
            stigmergy_average,
            stigmergy_std,
            random_average,
            random_std,
        ),
        xy=(
            COMMON_COMPARISON_TIME_SECONDS,
            stigmergy_mean_at_comparison,
        ),
        xytext=(18, 18),
        textcoords="offset points",
        fontsize=10,
        color="#9f2d0b",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#fff4ef",
            "edgecolor": "#d84315",
            "alpha": 0.95,
        },
        arrowprops={"arrowstyle": "->", "color": "#d84315"},
        zorder=7,
    )
    axis.text(
        0.015,
        0.975,
        "Bold lines: method means\n"
        "Thin lines: individual runs\n"
        "Shading: observed min-max range",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="0.25",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=True,
    )

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    print("Saved coverage graph: {0}".format(output))
    print("Saved analysis table: {0}".format(csv_path))

    for run in [centralized] + random_runs + stigmergy_runs:
        print(
            "{0}: final={1:.2f}% after {2:.1f}s, mean coverage/AUC={3:.2f}% "
            "({4})".format(
                run["path"],
                run["coverage"][-1],
                run["time"][-1],
                normalized_auc(run),
                run["source"],
            )
        )

    print(
        "Common 200 s comparison: stigmergy={0:.2f}% (SD {1:.2f}), "
        "random walk={2:.2f}% (SD {3:.2f}), difference={4:+.2f} points.".format(
            stigmergy_average,
            stigmergy_std,
            random_average,
            random_std,
            advantage,
        )
    )
    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot centralized, random-walk, and stigmergy coverage over time."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root containing centralized_results, random_walk_results, and "
            "stigmergy_results (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=("E1", "E2"),
        default=DEFAULT_EXPERIMENT,
        help="Experiment branch to discover (default: %(default)s)",
    )
    parser.add_argument(
        "--centralized",
        type=Path,
        default=None,
        help="Override the discovered centralized run or metadata.json path",
    )
    parser.add_argument(
        "--random",
        type=Path,
        nargs="+",
        default=None,
        metavar="RUN",
        help="Override the discovered random-walk runs or metadata.json paths",
    )
    parser.add_argument(
        "--stigmergy",
        type=Path,
        nargs="+",
        default=None,
        metavar="RUN",
        help="Override the discovered stigmergy runs or metadata.json paths",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (default: <results root>/"
            "coverage_over_time_<experiment>.png)"
        ),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=(
            "Output CSV path (default: <results root>/"
            "coverage_analysis_<experiment>.csv)"
        ),
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULT_CELL_SIZE_METERS,
        help="Grid size used when coverage must be reconstructed",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the graph without opening a Matplotlib window",
    )
    args = parser.parse_args()
    if args.cell_size <= 0.0:
        parser.error("--cell-size must be greater than zero")
    return args


def main():
    args = parse_args()
    discovered = None
    if args.centralized is None or args.random is None or args.stigmergy is None:
        discovered = discover_experiment_runs(args.results_root, args.experiment)
    centralized_path = args.centralized or discovered["centralized"][0]
    random_paths = args.random or discovered["random"]
    stigmergy_paths = args.stigmergy or discovered["stigmergy"]
    output = args.output or (
        args.results_root / "coverage_over_time_{0}.png".format(args.experiment)
    )
    csv_output = args.csv_output or (
        args.results_root / "coverage_analysis_{0}.csv".format(args.experiment)
    )
    create_plot(
        centralized_path=centralized_path,
        random_paths=random_paths,
        stigmergy_paths=stigmergy_paths,
        output=output,
        csv_output=csv_output,
        cell_size=args.cell_size,
        show=not args.no_show,
        experiment=args.experiment,
    )


if __name__ == "__main__":
    main()
