"""Compare E1 and E2 experiment results for the three search algorithms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
Z_95 = 1.96
FAILURE_MODES = ["early", "middle", "late", "mixed"]
ALGORITHM_LABELS = {
    "centralized": "Centralized",
    "stigmergy_random_walk": "Stigmergy Random Walk",
    "stigmergy_search": "Stigmergy Efficient Search",
}
ALGORITHM_ORDER = list(ALGORITHM_LABELS)
ALGORITHM_COLORS = {
    "centralized": "#4c78a8",
    "stigmergy_random_walk": "#f58518",
    "stigmergy_search": "#54a24b",
}
RESULT_FILES = {
    "centralized": {
        "E1": PROJECT_ROOT / "centralized" / "experiment_results" / "E1" / "results.xlsx",
        "E2": PROJECT_ROOT / "centralized" / "experiment_results" / "E2" / "results.xlsx",
    },
    "stigmergy_random_walk": {
        "E1": PROJECT_ROOT / "stigmergy_random_walk" / "experiment_results_stigmergy" / "E1" / "results.xlsx",
        "E2": PROJECT_ROOT / "stigmergy_random_walk" / "experiment_results_stigmergy" / "E2" / "results.xlsx",
    },
    "stigmergy_search": {
        "E1": PROJECT_ROOT / "stigmergy_search" / "experiment_results_stigmergy_efficient" / "E1" / "results.xlsx",
        "E2": PROJECT_ROOT / "stigmergy_search" / "experiment_results_stigmergy_efficient" / "E2" / "results.xlsx",
    },
}


def _horizon(grid_size: pd.Series, n_robots: pd.Series) -> pd.Series:
    return np.ceil(10.0 * (grid_size.astype(float) ** 2) / n_robots.astype(float))


def load_results() -> pd.DataFrame:
    frames = []
    for algorithm, experiments in RESULT_FILES.items():
        for experiment, path in experiments.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {experiment} workbook for {algorithm}: {path}")

            frame = pd.read_excel(path, sheet_name="detailed")
            frame["algorithm"] = algorithm
            frame["algorithm_label"] = ALGORITHM_LABELS[algorithm]
            frame["experiment"] = experiment
            frame["simulation_id"] = frame.get("simulation_id", 0)
            frame["failure_time_mode"] = frame.get("failure_time_mode", "no_failure")
            frame["horizon"] = _horizon(frame["grid_size"], frame["n_robots"])
            frame["target_found_fraction"] = frame["n_targets_found"] / frame["n_targets"]
            frame["coverage_fraction"] = frame["percent_coverage"] / 100.0
            frame["t_targets_horizon_fraction"] = frame["t_targets"] / frame["horizon"]
            frame["t_coverage_horizon_fraction"] = frame["t_coverage"] / frame["horizon"]
            frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def mean_ci_summary(data: pd.DataFrame, groups: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["runs"] = len(group)

        for metric in metrics:
            values = group[metric].dropna().astype(float)
            mean = values.mean()
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = Z_95 * sem
        rows.append(row)
    return pd.DataFrame(rows)


def pair_failure_effects(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["experiment"] == "E1"].copy()
    failures = results[results["experiment"] == "E2"].copy()
    keys = ["algorithm", "grid_size", "n_robots", "n_targets", "experiment_id", "simulation_id"]
    metrics = [
        "t_targets",
        "t_coverage",
        "t_targets_horizon_fraction",
        "t_coverage_horizon_fraction",
        "target_found_fraction",
        "coverage_fraction",
        "mean_found",
    ]

    baseline_cols = keys + metrics
    effects = failures.merge(
        baseline[baseline_cols],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("_e2", "_e1"),
    )
    if effects[[f"{metric}_e1" for metric in metrics]].isna().any().any():
        raise ValueError("Some E2 rows could not be paired with an E1 baseline run.")

    for metric in metrics:
        effects[f"delta_{metric}"] = effects[f"{metric}_e2"] - effects[f"{metric}_e1"]
        if metric.startswith("t_"):
            effects[f"pct_change_{metric}"] = (
                effects[f"delta_{metric}"] / effects[f"{metric}_e1"].replace(0, np.nan)
            ) * 100.0

    return effects


def _prepare_figure_dir(figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for old_figure in figure_dir.glob("*.png"):
        old_figure.unlink()


def _grouped_bar_plot(
    summary: pd.DataFrame,
    category_col: str,
    series_col: str,
    metric: str,
    category_order: list,
    series_order: list[str],
    series_labels: dict[str, str],
    series_colors: dict[str, str],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    xlabel: str = "",
    log_y: bool = False,
    zero_line: bool = False,
    note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(category_order), dtype=float)
    width = min(0.78 / max(len(series_order), 1), 0.26)

    for index, series in enumerate(series_order):
        group = (
            summary[summary[series_col] == series]
            .set_index(category_col)
            .reindex(category_order)
        )
        offset = (index - (len(series_order) - 1) / 2) * width
        ax.bar(
            x + offset,
            group[f"{metric}_mean"],
            width,
            yerr=group[f"{metric}_ci95"],
            capsize=3,
            label=series_labels.get(series, str(series)),
            color=series_colors.get(series),
            edgecolor="white",
            linewidth=0.7,
        )

    if log_y:
        ax.set_yscale("log")
    if zero_line:
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.65)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(value).title() for value in category_order])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    if note:
        ax.text(0.01, -0.18, note, transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _paired_search_effects(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot_table(
        index=["experiment", "grid_size"],
        columns="algorithm",
        values="t_targets",
        aggfunc="mean",
    ).reset_index()

    rows = []
    for baseline in ["centralized", "stigmergy_random_walk"]:
        comparison = pivot.dropna(subset=[baseline, "stigmergy_search"]).copy()
        comparison["baseline_algorithm"] = baseline
        comparison["search_target_time_reduction_pct"] = (
            (comparison[baseline] - comparison["stigmergy_search"]) / comparison[baseline].replace(0, np.nan)
        ) * 100.0
        comparison["search_target_time_speedup"] = (
            comparison[baseline] / comparison["stigmergy_search"].replace(0, np.nan)
        )
        rows.append(comparison)
    return pd.concat(rows, ignore_index=True)


def plot_key_figures(results: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    metrics = [
        "t_targets",
    ]
    grid_summary = mean_ci_summary(
        results,
        ["experiment", "algorithm", "grid_size"],
        metrics,
    )
    grid_order = sorted(results["grid_size"].unique())

    for experiment in ["E1", "E2"]:
        subset = grid_summary[grid_summary["experiment"] == experiment]
        _grouped_bar_plot(
            subset,
            "grid_size",
            "algorithm",
            "t_targets",
            grid_order,
            ALGORITHM_ORDER,
            ALGORITHM_LABELS,
            ALGORITHM_COLORS,
            f"{experiment}: Steps to Find All Targets",
            "Steps to all targets found (log scale)",
            output_dir / f"{experiment.lower()}_01_target_time_raw_log_all_algorithms.png",
            xlabel="Grid size",
            log_y=True,
            note=(
                "Lower is better. Log scale keeps all three algorithms readable despite random walk's larger times. "
                "Error bars: 95% CI over runs at each grid size."
            ),
        )

    e1_subset = grid_summary[grid_summary["experiment"] == "E1"]
    _grouped_bar_plot(
        e1_subset,
        "grid_size",
        "algorithm",
        "t_targets",
        grid_order,
        ["centralized", "stigmergy_search"],
        ALGORITHM_LABELS,
        ALGORITHM_COLORS,
        "E1: Centralized vs Stigmergy Efficient Search",
        "Steps to all targets found",
        output_dir / "e1_02_centralized_vs_stigmergy_search_target_time.png",
        xlabel="Grid size",
        note=(
            "Focused linear-scale view removes random walk so the efficient-search gap against centralized is visible. "
            "Error bars: 95% CI."
        ),
    )

    search_effects = _paired_search_effects(results)
    search_summary = mean_ci_summary(
        search_effects,
        ["experiment", "baseline_algorithm"],
        ["search_target_time_reduction_pct", "search_target_time_speedup"],
    )

    failure_by_mode = mean_ci_summary(
        effects,
        ["failure_time_mode", "algorithm"],
        [
            "delta_t_targets",
            "pct_change_t_targets",
        ],
    )
    _grouped_bar_plot(
        failure_by_mode,
        "failure_time_mode",
        "algorithm",
        "pct_change_t_targets",
        FAILURE_MODES,
        ALGORITHM_ORDER,
        ALGORITHM_LABELS,
        ALGORITHM_COLORS,
        "E2: Increase in Target Discovery Time with Failures",
        "Increase in time to discover all targets vs no failures (%)",
        output_dir / "e2_02_failure_effect_target_time_by_mode.png",
        xlabel="Failure timing",
        zero_line=True,
        note=(
            "Each bar is the paired percent increase from the matching no-failure E1 scenario. "
            "Positive values mean failures made target discovery take longer. "
            "Error bars: 95% CI."
        ),
    )

    return {
        "grid_plot_summary": grid_summary,
        "search_effect_summary": search_summary,
        "failure_plot_summary": failure_by_mode,
    }


def write_summaries(results: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    summaries = {
        "grid_performance": mean_ci_summary(
            results,
            ["experiment", "algorithm", "grid_size"],
            [
                "t_targets",
                "t_coverage",
                "mean_found",
                "target_found_fraction",
                "percent_coverage",
                "t_targets_horizon_fraction",
                "t_coverage_horizon_fraction",
            ],
        ),
        "overall_performance": mean_ci_summary(
            results,
            ["experiment", "algorithm"],
            [
                "t_targets_horizon_fraction",
                "t_coverage_horizon_fraction",
                "mean_found",
                "target_found_fraction",
                "percent_coverage",
            ],
        ),
        "failure_mode_effect": mean_ci_summary(
            effects,
            ["algorithm", "failure_time_mode"],
            [
                "delta_t_targets",
                "delta_t_coverage",
                "pct_change_t_targets",
                "pct_change_t_coverage",
                "delta_mean_found",
                "delta_target_found_fraction",
                "delta_coverage_fraction",
            ],
        ),
        "failure_grid_effect": mean_ci_summary(
            effects,
            ["algorithm", "failure_time_mode", "grid_size"],
            [
                "delta_t_targets",
                "delta_t_coverage",
                "delta_mean_found",
                "delta_target_found_fraction",
                "delta_coverage_fraction",
            ],
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "combined_detailed_results.csv", index=False)
    effects.to_csv(output_dir / "paired_e2_failure_effects.csv", index=False)
    for name, summary in summaries.items():
        summary.to_csv(output_dir / f"{name}.csv", index=False)
    return summaries


def _format_metric(mean: float, ci95: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {ci95:.{digits}f}"


def print_data_audit(results: pd.DataFrame, effects: pd.DataFrame) -> None:
    print("\nDATA AUDIT")
    counts = results.groupby(["experiment", "algorithm"]).size().unstack(fill_value=0)
    print(counts.rename(columns=ALGORITHM_LABELS).to_string())
    print("\nE2 failure timing rows")
    mode_counts = (
        results[results["experiment"] == "E2"]
        .groupby(["algorithm", "failure_time_mode"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=FAILURE_MODES, fill_value=0)
    )
    print(mode_counts.rename(index=ALGORITHM_LABELS).to_string())
    print(f"\nPaired E2 rows matched to E1 baseline: {len(effects):,}")


def print_overall_summary(overall: pd.DataFrame) -> None:
    view = overall.copy()
    display = pd.DataFrame({
        "Experiment": view["experiment"],
        "Algorithm": view["algorithm"].map(ALGORITHM_LABELS),
        "Runs": view["runs"],
        "Target time / horizon": [
            _format_metric(mean, ci) for mean, ci in zip(
                view["t_targets_horizon_fraction_mean"], view["t_targets_horizon_fraction_ci95"]
            )
        ],
        "Coverage time / horizon": [
            _format_metric(mean, ci) for mean, ci in zip(
                view["t_coverage_horizon_fraction_mean"], view["t_coverage_horizon_fraction_ci95"]
            )
        ],
        "Mean found": [
            _format_metric(mean, ci) for mean, ci in zip(view["mean_found_mean"], view["mean_found_ci95"])
        ],
        "Targets final": [
            _format_metric(mean, ci) for mean, ci in zip(
                view["target_found_fraction_mean"], view["target_found_fraction_ci95"]
            )
        ],
        "Coverage final %": [
            _format_metric(mean, ci, digits=2) for mean, ci in zip(
                view["percent_coverage_mean"], view["percent_coverage_ci95"]
            )
        ],
    })
    print("\nOVERALL ALGORITHM PERFORMANCE")
    print(display.to_string(index=False))


def print_failure_summary(failure_summary: pd.DataFrame) -> None:
    ordered = failure_summary.copy()
    ordered["failure_time_mode"] = pd.Categorical(ordered["failure_time_mode"], FAILURE_MODES, ordered=True)
    ordered = ordered.sort_values(["algorithm", "failure_time_mode"])
    display = pd.DataFrame({
        "Algorithm": ordered["algorithm"].map(ALGORITHM_LABELS),
        "Mode": ordered["failure_time_mode"].astype(str),
        "Runs": ordered["runs"],
        "Target step delta": [
            _format_metric(mean, ci, digits=1) for mean, ci in zip(
                ordered["delta_t_targets_mean"], ordered["delta_t_targets_ci95"]
            )
        ],
        "Target slowdown %": [
            _format_metric(mean, ci, digits=1) for mean, ci in zip(
                ordered["pct_change_t_targets_mean"], ordered["pct_change_t_targets_ci95"]
            )
        ],
        "Coverage step delta": [
            _format_metric(mean, ci, digits=1) for mean, ci in zip(
                ordered["delta_t_coverage_mean"], ordered["delta_t_coverage_ci95"]
            )
        ],
        "Mean found delta": [
            _format_metric(mean, ci) for mean, ci in zip(
                ordered["delta_mean_found_mean"], ordered["delta_mean_found_ci95"]
            )
        ],
    })
    print("\nPAIRED E2 FAILURE EFFECTS AGAINST E1")
    print(display.to_string(index=False))


def print_insights(
    overall: pd.DataFrame,
    failure_summary: pd.DataFrame,
    search_summary: pd.DataFrame,
) -> None:
    print("\nAUTO-GENERATED INSIGHTS")
    for experiment in ["E1", "E2"]:
        subset = overall[overall["experiment"] == experiment]
        fastest_targets = subset.loc[subset["t_targets_horizon_fraction_mean"].idxmin()]
        fastest_coverage = subset.loc[subset["t_coverage_horizon_fraction_mean"].idxmin()]
        best_found = subset.loc[subset["mean_found_mean"].idxmax()]
        print(
            f"- {experiment}: fastest normalized target completion is "
            f"{ALGORITHM_LABELS[fastest_targets['algorithm']]} "
            f"({fastest_targets['t_targets_horizon_fraction_mean']:.3f} of horizon on average)."
        )
        print(
            f"- {experiment}: fastest normalized coverage completion is "
            f"{ALGORITHM_LABELS[fastest_coverage['algorithm']]} "
            f"({fastest_coverage['t_coverage_horizon_fraction_mean']:.3f} of horizon on average)."
        )
        print(
            f"- {experiment}: highest mean target discovery over the run is "
            f"{ALGORITHM_LABELS[best_found['algorithm']]} "
            f"({best_found['mean_found_mean']:.3f})."
        )

    print("\nSTIGMERGY EFFICIENT SEARCH TARGET-TIME EFFECT")
    ordered_search = search_summary.sort_values(["experiment", "baseline_algorithm"])
    for _, row in ordered_search.iterrows():
        reduction = row["search_target_time_reduction_pct_mean"]
        ci95 = row["search_target_time_reduction_pct_ci95"]
        speedup = row["search_target_time_speedup_mean"]
        baseline_label = ALGORITHM_LABELS[row["baseline_algorithm"]]
        if reduction >= 0:
            effect_text = f"reduces target-completion time by {reduction:.1f}% +/- {ci95:.1f}%"
        else:
            effect_text = f"increases target-completion time by {abs(reduction):.1f}% +/- {ci95:.1f}%"
        print(
            f"- {row['experiment']}: compared with {baseline_label}, "
            f"Stigmergy Efficient Search {effect_text} "
            f"(mean speed ratio across grid sizes: {speedup:.2f}x)."
        )

    for algorithm in ALGORITHM_LABELS:
        subset = failure_summary[failure_summary["algorithm"] == algorithm]
        target_worst = subset.loc[subset["delta_t_targets_mean"].idxmax()]
        found_worst = subset.loc[subset["delta_mean_found_mean"].idxmin()]
        print(
            f"- {ALGORITHM_LABELS[algorithm]}: the largest average E2 target-time increase is "
            f"{target_worst['failure_time_mode']} failure "
            f"({target_worst['delta_t_targets_mean']:.1f} extra steps)."
        )
        print(
            f"- {ALGORITHM_LABELS[algorithm]}: the most negative mean-found change is "
            f"{found_worst['failure_time_mode']} failure "
            f"({found_worst['delta_mean_found_mean']:.3f})."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    _prepare_figure_dir(figure_dir)

    results = load_results()
    effects = pair_failure_effects(results)
    plot_summaries = plot_key_figures(results, effects, figure_dir)
    summaries = write_summaries(results, effects, output_dir)

    # Keep plot summaries available as CSV alongside the broader tables.
    for name, plot_summary in plot_summaries.items():
        plot_summary.to_csv(output_dir / f"{name}.csv", index=False)

    print_data_audit(results, effects)
    print_overall_summary(summaries["overall_performance"])
    print_failure_summary(summaries["failure_mode_effect"])
    print_insights(
        summaries["overall_performance"],
        summaries["failure_mode_effect"],
        plot_summaries["search_effect_summary"],
    )
    print(f"\nSaved CSV summaries to: {output_dir}")
    print(f"Saved error-bar figures to: {figure_dir}")


if __name__ == "__main__":
    main()
