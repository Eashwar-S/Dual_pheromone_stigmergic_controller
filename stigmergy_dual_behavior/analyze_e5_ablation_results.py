"""Analyze E5 dual-behavior ablation results.

The figures are intentionally limited to three:
1. completion success by start-distance shell;
2. first-discovery versus all-robot completion rate;
3. efficiency on successful completed runs.
4. efficacy-efficiency ablation landscape.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for E5 analysis. Install requirements or run from an environment with matplotlib."
    ) from exc


CURRENT_DIR = Path(__file__).resolve().parent
E5_DIR = CURRENT_DIR / "E5"
OUTPUT_DIR = E5_DIR / "analysis_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
Z_95 = 1.96

VARIANT_ORDER = [
    "repulsive_only",
    "attractive_only",
    "single_evaporative_merged",
    "single_persistent_merged",
    "sign_flip",
    "stigmergy_search_and_rescue",
]

VARIANT_LABELS = {
    "repulsive_only": "Repulsive-only",
    "attractive_only": "Attractive-only",
    "single_evaporative_merged": "Single evaporative\nmerged",
    "single_persistent_merged": "Single persistent\nmerged",
    "sign_flip": "Sign-flip",
    "stigmergy_search_and_rescue": "STIGSAR\nsearch + rescue",
}

COLORS = {
    "repulsive_only": "#4c78a8",
    "attractive_only": "#f58518",
    "single_evaporative_merged": "#b279a2",
    "single_persistent_merged": "#72b7b2",
    "sign_flip": "#e45756",
    "stigmergy_search_and_rescue": "#54a24b",
}

MARKERS = {
    "repulsive_only": "o",
    "attractive_only": "s",
    "single_evaporative_merged": "^",
    "single_persistent_merged": "D",
    "sign_flip": "v",
    "stigmergy_search_and_rescue": "*",
}

LINE_STYLES = {
    "repulsive_only": "-",
    "attractive_only": "--",
    "single_evaporative_merged": ":",
    "single_persistent_merged": "-.",
    "sign_flip": (0, (3, 1, 1, 1)),
    "stigmergy_search_and_rescue": "-",
}

X_OFFSETS = {
    "repulsive_only": -1.5,
    "attractive_only": -0.9,
    "single_evaporative_merged": -0.3,
    "single_persistent_merged": 0.3,
    "sign_flip": 0.9,
    "stigmergy_search_and_rescue": 1.5,
}


def ci95(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) <= 1:
        return 0.0
    return float(Z_95 * values.std(ddof=1) / np.sqrt(len(values)))


def mean_ci_summary(data: pd.DataFrame, groups: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby(groups, dropna=False, sort=True, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["runs"] = len(group)
        for metric in metrics:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_ci95"] = ci95(values)
        rows.append(row)
    return pd.DataFrame(rows)


def load_detailed_results() -> pd.DataFrame:
    rows = []
    missing = []
    for variant in VARIANT_ORDER:
        result_path = E5_DIR / f"{variant}_parallel_experiment" / "results.xlsx"
        if not result_path.exists():
            missing.append(result_path)
            continue
        frame = pd.read_excel(result_path, sheet_name="detailed")
        frame["variant"] = variant
        rows.append(frame)

    if not rows:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "No E5 result workbooks were found. Run one or more E5 parallel experiments first.\n"
            f"Expected files:\n{missing_list}"
        )

    data = pd.concat(rows, ignore_index=True)
    data["variant"] = pd.Categorical(data["variant"], categories=VARIANT_ORDER, ordered=True)
    data["completed_successfully"] = data["all_found"].astype(float)
    data["first_discovery_success"] = data["first_found"].astype(float)
    data["normalized_completion_cost"] = data["steps_to_all_found"] / data["optimal_manhattan_steps"].replace(0, np.nan)
    data.loc[data["completed_successfully"] < 1, "normalized_completion_cost"] = np.nan
    return data


def prepare_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old in FIGURE_DIR.glob("*.png"):
        old.unlink()


def safe_to_csv(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_new{path.suffix}")
        frame.to_csv(fallback, index=False)
        print(f"Could not overwrite locked file: {path}")
        print(f"Wrote fallback CSV instead: {fallback}")
        return fallback


def plot_completion_by_shell(data: pd.DataFrame, shell_summary: pd.DataFrame) -> Path:
    shells = sorted(data["requested_distance_shell"].unique())
    fig, ax = plt.subplots(figsize=(12.8, 6.6), constrained_layout=True)

    for variant in VARIANT_ORDER:
        line = shell_summary[shell_summary["variant"] == variant].sort_values("requested_distance_shell")
        if line.empty:
            continue
        x = line["requested_distance_shell"].to_numpy(dtype=float) + X_OFFSETS[variant]
        y = line["completed_successfully_mean"].to_numpy(dtype=float) * 100.0
        ci = line["completed_successfully_ci95"].fillna(0.0).to_numpy(dtype=float) * 100.0
        ax.plot(
            x,
            y,
            marker=MARKERS[variant],
            linestyle=LINE_STYLES[variant],
            linewidth=2.4 if variant == "stigmergy_search_and_rescue" else 1.9,
            markersize=9 if variant == "stigmergy_search_and_rescue" else 6,
            color=COLORS[variant],
            label=VARIANT_LABELS[variant].replace("\n", " "),
            zorder=4 if variant == "stigmergy_search_and_rescue" else 3,
        )
        ax.fill_between(x, np.maximum(y - ci, 0.0), np.minimum(y + ci, 100.0), color=COLORS[variant], alpha=0.12)
        end_y = y[-1]
        ax.text(
            x[-1] + 0.9,
            end_y + (X_OFFSETS[variant] * 0.6),
            VARIANT_LABELS[variant].replace("\n", " "),
            color=COLORS[variant],
            fontsize=8.5,
            va="center",
        )

    ax.set_title("E5 Completion Success by Start Distance", fontsize=15, pad=12)
    ax.set_xlabel("Requested Manhattan start-distance shell", fontsize=12)
    ax.set_ylabel("All robots found target (%)", fontsize=12)
    ax.set_xticks(shells)
    ax.set_xlim(min(shells) - 4.0, max(shells) + 14.0)
    ax.set_ylim(-2, 102)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9)

    output_path = FIGURE_DIR / "01_completion_success_by_distance.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_discovery_vs_completion(overall: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    x = np.arange(len(VARIANT_ORDER), dtype=float)
    width = 0.36

    overall = overall.set_index("variant").reindex(VARIANT_ORDER)
    first = overall["first_discovery_success_mean"].to_numpy(dtype=float) * 100.0
    first_ci = overall["first_discovery_success_ci95"].fillna(0.0).to_numpy(dtype=float) * 100.0
    complete = overall["completed_successfully_mean"].to_numpy(dtype=float) * 100.0
    complete_ci = overall["completed_successfully_ci95"].fillna(0.0).to_numpy(dtype=float) * 100.0

    ax.bar(x - width / 2, first, width, yerr=first_ci, capsize=3, color="#9ecae9", label="At least one robot finds target")
    ax.bar(x + width / 2, complete, width, yerr=complete_ci, capsize=3, color="#31a354", label="All robots find target")

    ax.set_title("E5 Search Alone vs Search-and-Rescue Completion", fontsize=15, pad=12)
    ax.set_ylabel("Run success (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    output_path = FIGURE_DIR / "02_discovery_vs_completion.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_efficiency(overall: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    overall = overall.set_index("variant").reindex(VARIANT_ORDER)
    x = np.arange(len(VARIANT_ORDER), dtype=float)

    mean = overall["normalized_completion_cost_mean"].to_numpy(dtype=float)
    ci = overall["normalized_completion_cost_ci95"].fillna(0.0).to_numpy(dtype=float)
    bars = ax.bar(
        x,
        mean,
        yerr=ci,
        capsize=3,
        color=[COLORS[v] for v in VARIANT_ORDER],
        edgecolor="white",
        linewidth=0.8,
    )

    for bar, variant in zip(bars, VARIANT_ORDER):
        value = overall.loc[variant, "normalized_completion_cost_mean"]
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}x",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("E5 Efficiency on Completed Runs", fontsize=15, pad=12)
    ax.set_ylabel("Steps to all-found / shortest initial Manhattan distance", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ymax = np.nanmax(mean + ci) if np.isfinite(mean + ci).any() else 1.0
    ax.set_ylim(0, ymax * 1.18)
    ax.text(
        0.01,
        0.98,
        fill("Lower is better. Bars are computed only over runs where all robots reached the target.", 78),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )

    output_path = FIGURE_DIR / "03_efficiency_completed_runs.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ablation_landscape(overall: pd.DataFrame) -> Path:
    landscape = overall.set_index("variant").reindex(VARIANT_ORDER).copy()
    x = landscape["completed_successfully_mean"].to_numpy(dtype=float) * 100.0
    y = landscape["normalized_completion_cost_mean"].to_numpy(dtype=float)
    delay = landscape["rescue_delay_mean"].to_numpy(dtype=float)
    delay = np.nan_to_num(delay, nan=0.0)
    size = 180.0 + 520.0 * (delay / max(delay.max(), 1.0))

    fig, ax = plt.subplots(figsize=(9.8, 7.0), constrained_layout=True)
    ax.axvspan(95, 101, color="#eef8ee", zorder=0)
    ax.axhspan(1, 70, color="#eef8ee", zorder=0)
    ax.axvline(95, color="#bdbdbd", linewidth=1.0, linestyle="--", zorder=1)
    ax.axhline(100, color="#bdbdbd", linewidth=1.0, linestyle="--", zorder=1)

    for variant, xi, yi, si in zip(VARIANT_ORDER, x, y, size):
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        ax.scatter(
            xi,
            yi,
            s=si,
            color=COLORS[variant],
            marker=MARKERS[variant],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
            zorder=3,
        )
        label = VARIANT_LABELS[variant].replace("\n", " ")
        dx, dy = {
            "repulsive_only": (-9, 14),
            "attractive_only": (3, 0),
            "single_evaporative_merged": (-21, 0),
            "single_persistent_merged": (-22, -14),
            "sign_flip": (3, -10),
            "stigmergy_search_and_rescue": (-22, -18),
        }[variant]
        ax.annotate(
            label,
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.2,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#9e9e9e", linewidth=0.7, shrinkA=0, shrinkB=5),
        )

    ax.set_title("E5 Ablation Landscape: Reliability, Cost, and Rescue Delay", fontsize=15, pad=12)
    ax.set_xlabel("All-robot completion success (%)", fontsize=12)
    ax.set_ylabel("Completion cost: steps / shortest initial distance (log scale)", fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(25, 103)
    finite_y = y[np.isfinite(y) & (y > 0)]
    ax.set_ylim(max(10, finite_y.min() * 0.55), finite_y.max() * 1.65)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.10)
    ax.text(
        0.03,
        0.04,
        "Best region: high completion, low cost.\nBubble area encodes first-to-all rescue delay.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#dddddd", alpha=0.9),
    )

    handles = [
        ax.scatter([], [], s=180, color="#777777", edgecolor="white", linewidth=1.0, label="Shorter delay"),
        ax.scatter([], [], s=700, color="#777777", edgecolor="white", linewidth=1.0, label="Longer delay"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    output_path = FIGURE_DIR / "04_ablation_landscape_reliability_cost_delay.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    prepare_output_dirs()
    detailed = load_detailed_results()

    shell_summary = mean_ci_summary(
        detailed,
        ["variant", "requested_distance_shell"],
        ["completed_successfully", "first_discovery_success", "normalized_completion_cost"],
    )
    overall = mean_ci_summary(
        detailed,
        ["variant"],
        ["completed_successfully", "first_discovery_success", "normalized_completion_cost", "rescue_delay", "n_found"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_to_csv(detailed, OUTPUT_DIR / "e5_detailed_combined.csv")
    safe_to_csv(shell_summary, OUTPUT_DIR / "e5_summary_by_distance_shell.csv")
    safe_to_csv(overall, OUTPUT_DIR / "e5_overall_variant_summary.csv")

    figure_paths = [
        plot_completion_by_shell(detailed, shell_summary),
        plot_discovery_vs_completion(overall),
        plot_efficiency(overall),
        plot_ablation_landscape(overall),
    ]

    print(f"Loaded {len(detailed):,} E5 runs across {detailed['variant'].nunique()} variants.")
    print(f"Saved analysis outputs to: {OUTPUT_DIR}")
    print("Saved figures:")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
