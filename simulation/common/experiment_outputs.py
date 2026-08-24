from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = [
    "algorithm",
    "grid_size",
    "n_robots",
    "n_targets",
    "n_failures",
    "failure_time_mode",
    "num_experiments",
    "num_simulations",
    "experiment_id",
    "simulation_id",
]

TIMESERIES_VALUE_COLUMNS = [
    "coverage_cells",
    "coverage_fraction",
    "targets_found",
    "target_fraction",
    "active_robots",
    "avg_visits_per_covered_cell",
    "pct_revisited_cells",
]


def split_scalar_and_timeseries(
    rows: Iterable[Dict[str, object]],
    algorithm: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    scalar_rows: List[Dict[str, object]] = []
    timeseries_rows: List[Dict[str, object]] = []

    for row in rows:
        scalar = {k: v for k, v in row.items() if k != "_time_series"}
        scalar["algorithm"] = algorithm
        scalar_rows.append(scalar)

        identity = {
            key: scalar[key]
            for key in IDENTITY_COLUMNS
            if key in scalar
        }
        for sample in row.get("_time_series", []) or []:
            timeseries_rows.append({**identity, **sample})

    return scalar_rows, timeseries_rows


def write_timeseries_csvs(
    timeseries_rows: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    ts_df = pd.DataFrame(timeseries_rows)
    sampled_path = output_dir / "timeseries_sampled.csv"
    summary_path = output_dir / "timeseries_summary.csv"

    ts_df.to_csv(sampled_path, index=False)
    if ts_df.empty:
        pd.DataFrame().to_csv(summary_path, index=False)
        return

    group_cols = [
        col
        for col in [
            "algorithm",
            "grid_size",
            "n_robots",
            "n_targets",
            "n_failures",
            "failure_time_mode",
            "sample_idx",
            "time_fraction",
        ]
        if col in ts_df.columns
    ]
    value_cols = [col for col in TIMESERIES_VALUE_COLUMNS if col in ts_df.columns]
    grouped = ts_df.groupby(group_cols, as_index=False)[value_cols].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join([part for part in col if part])
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]

    for value_col in value_cols:
        std_col = f"{value_col}_std"
        count_col = f"{value_col}_count"
        sem_col = f"{value_col}_sem"
        if std_col in grouped.columns and count_col in grouped.columns:
            grouped[sem_col] = grouped[std_col] / np.sqrt(grouped[count_col].clip(lower=1))

    grouped.to_csv(summary_path, index=False)


def autosize_excel_columns(writer, frames: List[Tuple[str, pd.DataFrame]]) -> None:
    for sheet_name, frame in frames:
        ws = writer.sheets[sheet_name]
        for idx, col in enumerate(frame.columns, 1):
            max_len = max([len(str(x)) for x in frame[col].astype(str)] + [len(col)])
            ws.set_column(idx - 1, idx - 1, min(max_len + 2, 60))
