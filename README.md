# Swarm Collaboration via Stigmergy

This repository compares centralized swarm search with stigmergy-based decentralized search methods. The experiments measure how different robot coordination strategies perform across grid sizes, robot counts, target counts, and failure settings.

## Codebase Structure

The four main algorithm folders are:

- `centralized/`: centralized coordination approach. Robots receive planned coverage regions and follow generated search paths.
- `stigmergy_random_walk/`: decentralized stigmergy random walk approach. Robots move using pheromone traces and local information.
- `stigmergy_search/`: decentralized stigmergy search approach with a more directed pheromone-based search policy.
- `multi_agent_rl/`: multi-agent reinforcement learning approach. Robots use a trained DQN policy to choose actions from local observations in the grid environment.

Experiment entry points are the `_experiment.py` files inside each algorithm folder:

- `centralized/centralized_experiment.py`
- `stigmergy_random_walk/stigmergy_random_experiment.py`
- `stigmergy_search/stigmergy_search_efficient_experiment.py`
- `multi_agent_rl/multi_agent_rl_experiment.py`

Shared support code is organized as:

- `config_experiments.py`: shared experiment parameters, including grid sizes, robot counts, target counts, failure counts, seeds, and horizon calculation.
- `common/`: shared simulation, geometry, utility, communication, and visualization helpers.
- `stigmergy_common/`: shared pheromone utilities used by the stigmergy approaches.


## Setup

From the repository root, install the dependencies:

```bash
pip install -r requirements.txt
```

If you prefer an isolated environment:

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
- `RUNS_PER_SCENARIO`, `BASE_SEED`, `ROBOT_RADIUS`, and `KAPPA`: shared constants used by the experiment scripts.

By default, `FAILURE_COUNTS` is set to zero failures. To run failure experiments, update `FAILURE_COUNTS` in `config_experiments.py`.

## How to Run

Run commands from the repository root.

Centralized experiments:

```bash
cd centralized
python centralized_experiment.py
```

Stigmergy random walk experiments:

```bash
cd stigmergy_random_walk/
python stigmergy_random_experiment.py
```

Stigmergy search experiments:

```bash
cd stigmergy_search/
python stigmergy_search_efficient_experiment.py
```

Each experiment script also supports a visualization mode:

```bash
python centralized/centralized_experiment.py --visualize
python stigmergy_random_walk/stigmergy_random_experiment.py --visualize
python stigmergy_search/stigmergy_search_efficient_experiment.py --visualize
```

## Outputs

Each experiment writes Excel results with two sheets:

- `detailed`: one row per run or simulation.
- `summary`: aggregated averages grouped by grid size, robot count, and failure count.

The output folders are created by the experiment scripts:

- `centralized/experiment_results/E1/results.xlsx` or `centralized/experiment_results/E2/results.xlsx`
- `stigmergy_random_walk/experiment_results_stigmergy/E1/results.xlsx` or `stigmergy_random_walk/experiment_results_stigmergy/E2/results.xlsx`
- `stigmergy_search/experiment_results_stigmergy_efficient/E1/results.xlsx` or `stigmergy_search/experiment_results_stigmergy_efficient/E2/results.xlsx`

`E1` is used when the configured failure count is zero. `E2` is used when failures are configured.

The main metrics saved in the result files include:

- `n_targets_found`: number of targets discovered.
- `t_targets`: timestep when all targets are found, or the timeout horizon if not completed.
- `t_coverage`: timestep when full coverage is reached, or the timeout horizon if not completed.
- `percent_coverage`: final percentage of the grid covered.
- `mean_found`: average target discovery progress over time.
