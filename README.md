# Swarm Collaboration via Stigmergy

This repository contains physical RoboMaster search experiments comparing a
centralized coverage strategy with two decentralized strategies. The recorded
experiment data is in [`robomaster_results/`](robomaster_results/), and the
top-level plotting scripts visualize robot coverage from that data.

## Search approaches

The results compare three approaches:

- **Centralized:** the arena is partitioned and robots follow coordinated
  coverage routes. In the failure experiment, another active robot takes over
  the reachable remainder of the failed robot's route.
- **Random walk:** a decentralized baseline in which each robot explores using
  random motion decisions rather than a shared coverage plan.
- **Stigmergy search:** decentralized robots use indirect coverage cues to
  prefer less-visited space, allowing their behavior to coordinate without a
  central controller.

## RoboMaster results

`robomaster_results/` is grouped first by approach and then by experiment:

```text
robomaster_results/
├── centralized_results/
│   ├── E1/run_1/
│   └── E2/E2_run_1/
├── random_walk_results/
│   ├── E1/run_1, run_2, run_3/
│   └── E2/E2_run_1, E2_run_2, E2_run_3/
└── stigmergy_results/
    ├── E1/run_1, run_2, run_3/
    └── E2/E2_run_1, E2_run_2, E2_run_3/
```

The two search experiments are:

- **E1 — without failures:** the robots complete the search under normal
  operation.
- **E2 — with failures:** one robot is stopped during the run. The remaining
  robots continue searching; the centralized approach explicitly reassigns
  reachable unfinished work, while the decentralized approaches continue
  according to their own search behavior.

Every run has a `metadata.json` file containing timestamps, robot positions,
actions, coverage data when recorded, and failure information for E2. When
camera recording was enabled, JPEG frames are stored below that same run in
`robot_1/`, `robot_2/`, or `robot_3/`. Some runs use metadata-only recording,
so these robot image folders are optional.

## Plotting the results

Install the plotting dependencies:

```bash
pip install -r requirements.txt
```

Plot the final swept coverage and redundant coverage for one run:

```bash
python plot_robot_coverage.py robomaster_results/stigmergy_results/E1/run_1 --no-show
```

The input can be a run directory, its `metadata.json`, or one of its
`robot_N/` frame directories. With `--no-show`, the live animation is skipped
and only the completed coverage figure is displayed. Close that window to save
`coverage_final.png` in the run directory. For E2 runs, the failed robot is
held at its failure position through the end of the visualization. Its movement
into the failure position is counted once, but its stationary footprint does
not add any later heat-map visits.

Compare all three approaches for E1 or E2:

```bash
python plot_coverage_over_time.py --experiment E1 --no-show
python plot_coverage_over_time.py --experiment E2 --no-show
```

The script discovers the available runs in `robomaster_results/` and writes
`coverage_over_time_E1.png`/`coverage_over_time_E2.png` plus the corresponding
`coverage_analysis_E1.csv`/`coverage_analysis_E2.csv` files in that directory.
Use `--results-root` for a results tree stored elsewhere, or the
`--centralized`, `--random`, and `--stigmergy` options to override individual
run paths.
