import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Optional

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent / "experimentation" / "experiment_results"
OUT  = Path(__file__).parent

# E1 files
E1 = {
    "centralized":  BASE / "centralized"              / "E1" / "results_e1.xlsx",
    "search_eff":   BASE / "stigmergy_search_efficient"/ "E1" / "results_e1.xlsx",
    # "random":       BASE / "stigmergy_random"          / "E1" / "results_e1.xlsx",
}

# E2 files
E2 = {
    "centralized":  BASE / "centralized"              / "E2" / "results_e2.xlsx",
    "search_eff":   BASE / "stigmergy_search_efficient"/ "E2" / "results_e2.xlsx",
    # "random":       BASE / "stigmergy_random"          / "E2" / "results_e2.xlsx",
}

# Rendezvous files (detailed sheet)
RDV = {
    "centralized_rendezvous": BASE / "centralized_rendezvous" / "results.xlsx",
    "stigmergy_rendezvous":   BASE / "stigmergy_rendezvous"   / "results.xlsx",
}

# ── style ──────────────────────────────────────────────────────────────────────
STYLE = {
    "centralized": dict(color="#1f77b4", marker="o", linestyle="-",  label="Centralized"),
    "search_eff":  dict(color="#2ca02c", marker="s", linestyle="-",  label="Stigmergy Search Efficient (Ours)"),
    "random":      dict(color="#d62728", marker="^", linestyle="-",  label="Stigmergy Random"),
    "centralized_rendezvous": dict(color="#1f77b4", marker="o", linestyle="-",  label="Centralized Rendezvous"),
    "stigmergy_rendezvous":   dict(color="#ff7f0e", marker="D", linestyle="-",  label="Stigmergy Rendezvous (Ours)"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})


# ── helper: load detailed sheet and aggregate by grid_size ────────────────────
def load_summary_agg(path: Path) -> Optional[pd.DataFrame]:
    """Read detailed sheet, compute mean ± std of t_targets per grid_size."""
    if not path.exists():
        print(f"  [skip] not found: {path}")
        return None
    try:
        df = pd.read_excel(path, sheet_name="detailed")
        agg = df.groupby("grid_size")["t_targets"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0)  # std=0 for single-run (deterministic) algorithms
        return agg
    except Exception as e:
        print(f"  [error] {path}: {e}")
        return None


def load_detailed_t_targets(path: Path) -> Optional[pd.Series]:
    """Read detailed sheet, return t_targets series (one value per run)."""
    if not path.exists():
        print(f"  [skip] not found: {path}")
        return None
    try:
        df = pd.read_excel(path, sheet_name="detailed")
        return df["t_targets"].reset_index(drop=True)
    except Exception as e:
        print(f"  [error] {path}: {e}")
        return None


# ── helper: plot one algorithm line with error bars ───────────────────────────
def plot_algo(ax, agg: pd.DataFrame, key: str):
    s = STYLE[key]
    if key == "centralized":
        # Deterministic — single run, no error bars
        ax.plot(agg["grid_size"], agg["mean"],
                label=s["label"], color=s["color"],
                marker=s["marker"], linestyle=s["linestyle"],
                linewidth=1.8, markersize=7)
    else:
        # Stochastic — show ±std error bars across runs
        ax.errorbar(agg["grid_size"], agg["mean"], yerr=agg["std"],
                    label=s["label"], color=s["color"],
                    marker=s["marker"], linestyle=s["linestyle"],
                    capsize=5, linewidth=1.8, markersize=7)


# ── Graph 1: E1 — three algorithms, grid_size vs t_targets ────────────────────
def plot_e1():
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False
    for key, path in E1.items():
        agg = load_summary_agg(path)
        if agg is not None:
            plot_algo(ax, agg, key)
            any_data = True

    ax.set_xlabel("Grid Size (L×L)", fontsize=12)
    ax.set_ylabel("Timesteps to Find All Targets (t_targets)", fontsize=12)
    ax.set_title("E1 — Target Discovery Time vs Grid Size (No Failures)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = OUT / "plot_E1_t_targets_vs_grid_size_1.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
    return any_data


# ── Graph 2: E2 — three algorithms, grid_size vs t_targets ────────────────────
def plot_e2():
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False
    for key, path in E2.items():
        agg = load_summary_agg(path)
        if agg is not None:
            plot_algo(ax, agg, key)
            any_data = True

    ax.set_xlabel("Grid Size (L×L)", fontsize=12)
    ax.set_ylabel("Timesteps to Find All Targets (t_targets)", fontsize=12)
    ax.set_title("E2 — Target Discovery Time vs Grid Size (With Failures)", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = OUT / "plot_E2_t_targets_vs_grid_size_1.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
    return any_data


# ── helper: load both rendezvous columns from detailed sheet ──────────────────
def load_rendezvous_cols(path: Path, col: str) -> Optional[pd.Series]:
    """Read detailed sheet, return the named column as a series."""
    if not path.exists():
        print(f"  [skip] not found: {path}")
        return None
    try:
        df = pd.read_excel(path, sheet_name="detailed")
        if col not in df.columns:
            print(f"  [skip] column '{col}' missing in {path}")
            return None
        return df[col].reset_index(drop=True)
    except Exception as e:
        print(f"  [error] {path}: {e}")
        return None


def _plot_rendezvous_metric(col: str, ylabel: str, title: str, outname: str):
    """Generic helper: plot one rendezvous metric (col) per run for both algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    any_data = False

    for key, path in RDV.items():
        series = load_rendezvous_cols(path, col)
        if series is not None:
            s = STYLE[key]
            runs = range(1, len(series) + 1)
            ax.plot(list(runs), series.tolist(),
                    label=s["label"], color=s["color"],
                    marker=s["marker"], linestyle=s["linestyle"],
                    linewidth=1.8, markersize=7)
            any_data = True

    ax.set_xlabel("Run Index", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = OUT / outname
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
    return any_data


# ── Graph 3a: Rendezvous — t_first_detect per run ─────────────────────────────
def plot_rendezvous_first_detect():
    return _plot_rendezvous_metric(
        col="t_first_detect",
        ylabel="t_first_detect (first robot locates target)",
        title="Rendezvous — First Detection Time per Run\n(centralized vs stigmergy, same positions per run)",
        outname="plot_rendezvous_t_first_detect_per_run.png",
    )


# ── Graph 3b: Rendezvous — t_targets per run ──────────────────────────────────
def plot_rendezvous_t_targets():
    return _plot_rendezvous_metric(
        col="t_targets",
        ylabel="t_targets (both robots have detected target)",
        title="Rendezvous — Rendezvous Complete Time per Run\n(centralized vs stigmergy, same positions per run)",
        outname="plot_rendezvous_t_targets_per_run.png",
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Plot 1: E1 ===")
    plot_e1()
    
    print("=== Plot 2: E2 ===")
    plot_e2()
    # print("=== Plot 3a: Rendezvous — t_first_detect ===")
    # plot_rendezvous_first_detect()
    print("=== Plot 3b: Rendezvous — t_targets ===")
    plot_rendezvous_t_targets()
    print("Done.")


if __name__ == "__main__":
    main()
