# Swarm Collaboration via Stigmergy

This repository compares centralized swarm search with decentralized search methods across grid sizes, robot counts, target counts, and failure settings.

## Codebase Structure

The three main algorithm folders are:

- `centralized/`: centralized coordination. Robots receive planned coverage regions and follow generated search paths.
- `random_walk/`: decentralized memoryless random walk baseline. Robots move randomly while respecting collision constraints.
- `stigmergy_search/`: decentralized stigmergy search with pheromone-guided movement.

Shared support code is organized as:

- `config_experiments.py`: shared experiment parameters, including grid sizes, robot counts, target counts, failure counts, seeds, and horizon calculation.
- `common/`: shared simulation, geometry, metric, output, utility, and visualization helpers.
- `stigmergy_common/`: shared pheromone utilities used by stigmergy-based approaches.

Important experiment entry points are:

- `centralized/centralized_parallel_experiment.py`
- `random_walk/random_walk_parallel_experiment.py`
- `stigmergy_search/stigmergy_search_efficient_parallel_experiment.py`

The non-parallel `_experiment.py` files remain available for direct single-process runs and visualization support.

## Setup

From the repository root, install dependencies:

```bash
pip install -r requirements.txt
```

For an isolated environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure Experiments

Edit `config_experiments.py` to change the experiment settings.

Important parameters include:

- `GRID_SIZES`: grid dimensions used in the experiments.
- `ROBOT_COUNTS`: number of robots for each scenario.
- `TARGET_COUNTS`: number of targets for each scenario.
- `FAILURE_COUNTS`: number of robot failures for each scenario.
- `FAILURE_TIME_MODE`: default failure timing mode for non-parallel runs.
- `RUNS_PER_SCENARIO`, `BASE_SEED`, `ROBOT_RADIUS`, and `KAPPA`: shared constants used by experiment scripts.

Use E1 for no-failure baseline experiments:

```python
FAILURE_COUNTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Use E2 for failure robustness experiments by setting nonzero failure counts:

```python
FAILURE_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

The output folder is selected automatically. If the first configured scenario has `n_failures == 0`, results go to `E1`; otherwise they go to `E2`.

## Run Parallel Experiments

Run commands from the repository root.

Centralized:

```bash
python centralized/centralized_parallel_experiment.py --workers 8
```

Random walk:

```bash
python random_walk/random_walk_parallel_experiment.py --workers 8
```

Stigmergy search:

```bash
python stigmergy_search/stigmergy_search_efficient_parallel_experiment.py --workers 8
```

If `--workers` is omitted, each script defaults to the detected CPU count.

For E2, the parallel scripts run four failure timing modes automatically:

- `mixed`
- `early`
- `middle`
- `late`

## Visualization

Each algorithm also has a visualization path:

```bash
python centralized/centralized_experiment.py --visualize
python random_walk/random_walk_experiment.py --visualize
python stigmergy_search/stigmergy_search_efficient_experiment.py --visualize
```

## Output Storage

Parallel experiment outputs are written under the current working directory. When commands are run from the repository root, the output folders are:

- `experiment_results/E1/` or `experiment_results/E2/` for centralized.
- `experiment_results_stigmergy/E1/` or `experiment_results_stigmergy/E2/` for random walk.
- `experiment_results_stigmergy_efficient/E1/` or `experiment_results_stigmergy_efficient/E2/` for stigmergy search.

Each E1/E2 folder contains:

- `results.xlsx`: scalar run results and grouped summaries.
- `timeseries_sampled.csv`: graph-ready per-run sampled time-series data.
- `timeseries_summary.csv`: averaged graph-ready curves by scenario and time sample.

`results.xlsx` contains:

- `detailed`: one row per run or simulation.
- `summary`: grouped averages by algorithm, grid size, robot count, failure count, and failure timing mode.

## Metrics

Core scalar metrics include:

- `n_targets_found`: number of targets discovered by the end of the run.
- `t_targets`: timestep when all targets are found, or the timeout horizon if not completed.
- `t_coverage`: timestep when full coverage is reached, or the timeout horizon if not completed.
- `percent_coverage`: final percentage of the grid covered.
- `mean_found`: average target discovery fraction over executed timesteps.

Additional graph and robustness metrics include:

- `max_horizon`: timeout horizon for the scenario.
- `steps_executed`: number of simulated steps before completion or timeout.
- `final_coverage_cells`: number of grid cells covered by the end of the run.
- `final_coverage_fraction`: final covered-cell fraction.
- `final_targets_fraction`: final discovered-target fraction.
- `coverage_auc_executed`: average coverage fraction over executed timesteps.
- `target_auc_executed`: average target discovery fraction over executed timesteps.
- `coverage_auc_horizon_norm`: coverage AUC normalized over the full horizon, padding early-finished runs with final values.
- `target_auc_horizon_norm`: target discovery AUC normalized over the full horizon, padding early-finished runs with final values.
- `avg_visits_per_covered_cell`: average number of robot footprint observations for cells that were covered at least once.
- `extra_visits_per_covered_cell`: average redundant observations beyond the first visit for covered cells.
- `pct_revisited_cells`: percentage of grid cells observed more than once.
- `total_cell_observations`: total robot footprint observations across the grid.
- `success_targets`: `1` if all targets were found before timeout, otherwise `0`.
- `success_coverage`: `1` if full coverage was reached before timeout, otherwise `0`.

E2-specific fields include:

- `failure_time_mode`: failure timing group, such as `mixed`, `early`, `middle`, or `late`.
- `failed_robot_ids`: semicolon-separated failed robot IDs.
- `failure_steps`: semicolon-separated failure timesteps.
- `first_failure_step`: earliest failure timestep in the run.
- `last_failure_step`: latest failure timestep in the run.
- `targets_found_at_first_failure`: number of targets found when the first failure has taken effect.
- `coverage_fraction_at_first_failure`: coverage fraction when the first failure has taken effect.
- `post_failure_target_auc`: average target discovery fraction after the first failure.
- `post_failure_coverage_auc`: average coverage fraction after the first failure.

## Time-Series Format

`timeseries_sampled.csv` stores sampled curves in long format. Each run contributes `201` samples from `time_fraction = 0.0` to `time_fraction = 1.0`.

Important columns include:

- Run identity: `algorithm`, `grid_size`, `n_robots`, `n_targets`, `n_failures`, `failure_time_mode`, `experiment_id`, and `simulation_id`.
- Time identity: `sample_idx`, `step`, and `time_fraction`.
- Curve values: `coverage_cells`, `coverage_fraction`, `targets_found`, `target_fraction`, `active_robots`, `avg_visits_per_covered_cell`, and `pct_revisited_cells`.

`timeseries_summary.csv` groups the sampled curves by scenario and time sample. It stores mean, standard deviation, count, and standard error columns for graph creation.
