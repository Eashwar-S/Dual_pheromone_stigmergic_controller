from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def visit_metrics(visit_counts: np.ndarray, total_cells: int) -> Dict[str, float]:
    covered_cells = int(np.sum(visit_counts > 0))
    total_observations = int(np.sum(visit_counts))
    revisited_cells = int(np.sum(visit_counts > 1))

    if covered_cells > 0:
        avg_visits = total_observations / covered_cells
        extra_visits = (total_observations - covered_cells) / covered_cells
    else:
        avg_visits = 0.0
        extra_visits = 0.0

    return {
        "total_cell_observations": total_observations,
        "avg_visits_per_covered_cell": avg_visits,
        "extra_visits_per_covered_cell": extra_visits,
        "pct_revisited_cells": (revisited_cells / total_cells) * 100.0 if total_cells else 0.0,
    }


def scalar_metrics(
    *,
    max_horizon: int,
    steps_executed: int,
    total_cells: int,
    final_coverage_cells: int,
    n_targets: int,
    n_targets_found: int,
    auc_cov_sum: float,
    auc_found_sum: float,
    visit_counts: np.ndarray,
    t_targets: int,
    t_coverage: int,
    first_failure_step: Optional[int],
    last_failure_step: Optional[int],
    failed_robot_ids: List[int],
    failure_steps: List[int],
    targets_found_at_first_failure: Optional[int],
    coverage_fraction_at_first_failure: Optional[float],
    post_failure_cov_sum: float,
    post_failure_found_sum: float,
    post_failure_steps: int,
) -> Dict[str, object]:
    final_coverage_fraction = final_coverage_cells / total_cells if total_cells else 0.0
    final_targets_fraction = n_targets_found / n_targets if n_targets else 1.0

    executed_den = max(steps_executed, 1)
    padded_cov_sum = auc_cov_sum + max(0, max_horizon - steps_executed) * final_coverage_fraction
    padded_found_sum = auc_found_sum + max(0, max_horizon - steps_executed) * final_targets_fraction
    horizon_den = max(max_horizon, 1)

    out: Dict[str, object] = {
        "max_horizon": max_horizon,
        "steps_executed": steps_executed,
        "final_coverage_cells": final_coverage_cells,
        "final_coverage_fraction": final_coverage_fraction,
        "final_targets_fraction": final_targets_fraction,
        "coverage_auc_executed": auc_cov_sum / executed_den if steps_executed > 0 else 0.0,
        "target_auc_executed": auc_found_sum / executed_den if steps_executed > 0 else 0.0,
        "coverage_auc_horizon_norm": padded_cov_sum / horizon_den,
        "target_auc_horizon_norm": padded_found_sum / horizon_den,
        "success_targets": 1 if t_targets < max_horizon else 0,
        "success_coverage": 1 if t_coverage < max_horizon else 0,
        "failed_robot_ids": ";".join(map(str, failed_robot_ids)),
        "failure_steps": ";".join(map(str, failure_steps)),
        "first_failure_step": first_failure_step if first_failure_step is not None else "",
        "last_failure_step": last_failure_step if last_failure_step is not None else "",
        "targets_found_at_first_failure": (
            targets_found_at_first_failure if targets_found_at_first_failure is not None else ""
        ),
        "coverage_fraction_at_first_failure": (
            coverage_fraction_at_first_failure if coverage_fraction_at_first_failure is not None else ""
        ),
        "post_failure_target_auc": (
            post_failure_found_sum / post_failure_steps if post_failure_steps > 0 else ""
        ),
        "post_failure_coverage_auc": (
            post_failure_cov_sum / post_failure_steps if post_failure_steps > 0 else ""
        ),
    }
    out.update(visit_metrics(visit_counts, total_cells))
    return out


@dataclass
class TimeSeriesSampler:
    max_horizon: int
    n_samples: int = 201

    def __post_init__(self):
        self.n_samples = max(2, int(self.n_samples))
        self.sample_steps = np.rint(np.linspace(0, self.max_horizon, self.n_samples)).astype(int)
        self.next_idx = 0
        self.rows: List[Dict[str, object]] = []

    def record(
        self,
        *,
        step: int,
        coverage_cells: int,
        total_cells: int,
        targets_found: int,
        total_targets: int,
        active_robots: int,
        visit_counts: np.ndarray,
    ) -> None:
        coverage_fraction = coverage_cells / total_cells if total_cells else 0.0
        target_fraction = targets_found / total_targets if total_targets else 1.0
        redundancy = visit_metrics(visit_counts, total_cells)

        while self.next_idx < self.n_samples and step >= int(self.sample_steps[self.next_idx]):
            sample_idx = self.next_idx
            self.rows.append(
                {
                    "sample_idx": sample_idx,
                    "step": int(step),
                    "time_fraction": sample_idx / (self.n_samples - 1),
                    "coverage_cells": int(coverage_cells),
                    "coverage_fraction": coverage_fraction,
                    "targets_found": int(targets_found),
                    "target_fraction": target_fraction,
                    "active_robots": int(active_robots),
                    "avg_visits_per_covered_cell": redundancy["avg_visits_per_covered_cell"],
                    "pct_revisited_cells": redundancy["pct_revisited_cells"],
                }
            )
            self.next_idx += 1

    def finalize(
        self,
        *,
        step: int,
        coverage_cells: int,
        total_cells: int,
        targets_found: int,
        total_targets: int,
        active_robots: int,
        visit_counts: np.ndarray,
    ) -> List[Dict[str, object]]:
        self.record(
            step=step,
            coverage_cells=coverage_cells,
            total_cells=total_cells,
            targets_found=targets_found,
            total_targets=total_targets,
            active_robots=active_robots,
            visit_counts=visit_counts,
        )
        while self.next_idx < self.n_samples:
            sample_idx = self.next_idx
            self.record(
                step=max(self.max_horizon, step, int(self.sample_steps[sample_idx])),
                coverage_cells=coverage_cells,
                total_cells=total_cells,
                targets_found=targets_found,
                total_targets=total_targets,
                active_robots=active_robots,
                visit_counts=visit_counts,
            )
        return self.rows


def first_last_failures(norm_sched: List[Tuple[int, int]]) -> Tuple[List[int], List[int], Optional[int], Optional[int]]:
    failed_robot_ids = [rid for rid, _ in norm_sched]
    failure_steps = [st for _, st in norm_sched]
    return (
        failed_robot_ids,
        failure_steps,
        min(failure_steps) if failure_steps else None,
        max(failure_steps) if failure_steps else None,
    )
