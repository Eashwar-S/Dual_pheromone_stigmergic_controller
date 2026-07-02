# Swarm Collaboration via Stigmergy

This repository studies swarm search and rescue using centralized control, random baselines, and stigmergy-based coordination. The code is organized by experiment: E1, E2, E4, and E5.

## Setup

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Optional virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Shared Configuration

Most E1/E2 experiment settings live in `config_experiments.py`.

Important parameters:

- `GRID_SIZES`: grid dimensions.
- `ROBOT_COUNTS`: number of robots.
- `TARGET_COUNTS`: number of targets.
- `FAILURE_COUNTS`: number of failed robots.
- `RUNS_PER_SCENARIO`: repetitions per scenario.
- `BASE_SEED`: base random seed.
- `ROBOT_RADIUS`: sensing/deposition radius.
- `KAPPA`: timeout-horizon scaling factor.

If `FAILURE_COUNTS` is all zeros, E1 outputs are used. If failures are nonzero, E2 outputs are used.

## Algorithms

These algorithm names are used across experiments:

- **Centralized search** (`centralized/`): a coordinator partitions the grid and assigns planned coverage paths to robots.
- **Random walk** (`random_walk/`): decentralized baseline where robots move randomly while respecting collision constraints.
- **Stigmergy search** (`stigmergy_search/`): decentralized search using evaporating repulsive pheromone; robots avoid recently searched areas to reduce redundant coverage.
- **Stigmergy random walk** (`stigmergy_random_walk/`): ablation of stigmergy search with a more random movement policy.
- **Rendezvous convergence** (`stigmergy_convergence/`): attractive-pheromone convergence to a known target.
- **Dual-behavior search and rescue** (`stigmergy_dual_behavior/`): E5 ablations for separating search dispersion from rescue convergence.

## E1: No-Failure Search Benchmark

**Goal:** compare search performance when all robots remain active.

**Folders and algorithms:**

- `centralized/`: centralized search.
- `random_walk/`: random walk baseline.
- `stigmergy_search/`: stigmergy search.
- `stigmergy_random_walk/`: ablation for stigmergy search.

**Configuration:** set all failure counts to zero in `config_experiments.py`.

```python
FAILURE_COUNTS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Run commands:**

```bash
python centralized/centralized_parallel_experiment.py --workers 8
python random_walk/random_walk_parallel_experiment.py --workers 8
python stigmergy_search/stigmergy_search_efficient_parallel_experiment.py --workers 8
python stigmergy_random_walk/stigmergy_random_parallel_experiment.py --workers 8
```

**Main comparison:** target discovery, coverage, completion time, redundant visits, and success rate without robot failures.

## E2: Failure-Robustness Search Benchmark

**Goal:** compare how the same search algorithms handle robot failures.

**Folders and algorithms:**

- `centralized/`: centralized search.
- `random_walk/`: random walk baseline.
- `stigmergy_search/`: stigmergy search.
- `stigmergy_random_walk/`: ablation for stigmergy search.

**Configuration:** set nonzero failure counts in `config_experiments.py`.

```python
FAILURE_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

The E2 parallel scripts evaluate failure timing modes automatically:

- `early`
- `middle`
- `late`
- `mixed`

**Run commands:**

```bash
python centralized/centralized_parallel_experiment.py --workers 8
python random_walk/random_walk_parallel_experiment.py --workers 8
python stigmergy_search/stigmergy_search_efficient_parallel_experiment.py --workers 8
python stigmergy_random_walk/stigmergy_random_parallel_experiment.py --workers 8
```

**Main comparison:** post-failure target discovery, post-failure coverage, robustness to failure timing, and final success rate.

## E4: Rendezvous Convergence

**Goal:** test whether search robots can converge to a known target using an attractive pheromone trail.

**Folder:** `stigmergy_convergence/`

**Setup:** one advertising robot starts at the center target and deposits a non-evaporating attractive spiral pheromone. Search robots start away from the target, search with repulsive pheromone, and switch to following when they detect the attractive field.

**Run command:**

```bash
python stigmergy_convergence/rendezvous_parallel_experiment.py --workers 8
```

**Visualization:**

```bash
python stigmergy_convergence/rendezvous_visualization.py --search-robots 3 --distance-shell 40 --grid-size 100
```

**Main comparison:** convergence time, attractive detection time, distance-to-target curves, and path efficiency.

## E5: Dual-Behavior Search-and-Rescue Ablation

**Goal:** test whether search-and-rescue needs two distinct pheromone roles: evaporating repulsive search memory and persistent attractive rescue recruitment.

**Folder:** `stigmergy_dual_behavior/`

All E5 variants use the same grids, robot counts, distance shells, start seeds, and algorithm seeds for fair comparison. Outputs are stored under `stigmergy_dual_behavior/E5/`.

### E5 Variants

- **Repulsive-only** (`repulsive_only_parallel_experiment.py`): robots search individually using evaporating repulsive pheromone; no attractive recruitment.
- **Attractive-only** (`attractive_only_parallel_experiment.py`): robots use attractive/self-reinforcing search behavior; no rescue recruitment mode.
- **Single evaporative merged field** (`single_evaporative_merged_parallel_experiment.py`): one evaporating field is used for merged search signaling.
- **Single persistent merged field** (`single_persistent_merged_parallel_experiment.py`): one non-evaporating field is used for merged search signaling.
- **Sign-flip** (`sign_flip_parallel_experiment.py`): robots follow high search pheromone instead of avoiding it, testing whether repulsive sign is necessary.
- **STIGSAR / stigmergy search and rescue** (`stigmergy_search_and_rescue_parallel_experiment.py`): full dual behavior. Robots start in search mode; the first robot that finds the target advertises with persistent attractive pheromone while others converge.

**Run commands:**

```bash
python stigmergy_dual_behavior/repulsive_only_parallel_experiment.py --workers 8
python stigmergy_dual_behavior/attractive_only_parallel_experiment.py --workers 8
python stigmergy_dual_behavior/single_evaporative_merged_parallel_experiment.py --workers 8
python stigmergy_dual_behavior/single_persistent_merged_parallel_experiment.py --workers 8
python stigmergy_dual_behavior/sign_flip_parallel_experiment.py --workers 8
python stigmergy_dual_behavior/stigmergy_search_and_rescue_parallel_experiment.py --workers 8
```

**Visualization examples:**

```bash
python stigmergy_dual_behavior/repulsive_only_visualization.py --search-robots 3 --grid-size 100 --distance-shell 50
python stigmergy_dual_behavior/stigmergy_search_and_rescue_visualization.py --search-robots 3 --grid-size 100 --distance-shell 50
```

**Analysis:**

```bash
python stigmergy_dual_behavior/analyze_e5_ablation_results.py
```

Analysis outputs are written to:

```text
stigmergy_dual_behavior/E5/analysis_outputs/
```

**Main comparison:** all-robot completion success, first-discovery versus all-found completion, rescue delay, and normalized completion cost.

## Output Files

Parallel experiment folders typically contain:

- `results.xlsx`: scalar run results and grouped summaries.
- `search_robot_metrics.csv`: per-robot metrics for E4/E5.
- `search_robot_paths.csv`: per-step robot path data for E4/E5.
- `swarm_convergence.csv`: per-step swarm convergence data for E4/E5.
- `timeseries_sampled.csv` and `timeseries_summary.csv`: sampled E1/E2 time-series outputs.

Large E5 CSV/XLSX outputs are tracked with Git LFS when committed.

## Metrics Summary

Common E1/E2 metrics:

- targets found and target success rate.
- coverage fraction and coverage success rate.
- completion time and timeout horizon.
- redundant visits and revisit fraction.
- post-failure target/coverage metrics for E2.

E4/E5 metrics:

- `first_found`: whether any robot found the target.
- `all_found`: whether all robots found the target.
- `steps_to_all_found`: completion time for the team.
- `rescue_delay`: time between first target discovery and all robots reaching the target.
- `time_to_attractive_detection`: when a robot first senses attractive pheromone.
- `final_distance`: final Manhattan distance to target.
- `normalized_completion_cost`: completion steps divided by shortest initial Manhattan distance.
