# Dual Pheromone Stigmergic Controller

This repository studies whether stigmergy-based local communication can improve decentralized swarm search, rescue, and convergence behaviors compared with centralized planning and random-walk baselines.

## Repository Layout

- [simulation](simulation/) contains the local grid-world implementations, batch experiment scripts, visualizations, and generated simulation result files under each experiment folder.
- [robotarium](robotarium/) contains Georgia Tech Robotarium-compatible scripts, the vendored Robotarium Python simulator package, the summary workbook [robotarium_results.xlsx](robotarium/robotarium_results.xlsx), and the convergence demonstration video in [robotarium/convergence_video](robotarium/convergence_video/); full Robotarium run artifacts are archived in [Google Drive](https://drive.google.com/drive/folders/1P1_hQbMMIihzn0ZBpZkgzDJFcpdXOGGC?usp=sharing).
- [robomaster](robomaster/) contains DJI RoboMaster hardware scripts in [robomaster/scripts](robomaster/scripts/) and physical-run metadata/results; full RoboMaster run artifacts are archived in [Google Drive](https://drive.google.com/drive/folders/11PYkNSyzJfIDC1ALmiRF3ahNZJvkLHqX?usp=sharing).

## Installation

Use Python 3.10 or newer. The Robotarium Python simulator also lists Python 3.10+, NumPy, Matplotlib, and CVXOPT as requirements in its official setup/README, and the Robotarium project recommends prototyping locally before submitting code through the Georgia Tech Robotarium web interface.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The Robotarium simulator package is vendored in [robotarium/rps](robotarium/rps/). If imports such as `rps.robotarium` are not found, add the Robotarium folder to `PYTHONPATH` from the repository root before running Robotarium scripts:

```bash
# Windows PowerShell
$env:PYTHONPATH = "$PWD\robotarium;$env:PYTHONPATH"
# macOS/Linux
export PYTHONPATH="$PWD/robotarium:$PYTHONPATH"
```

The RoboMaster scripts require access to DJI RoboMaster robots, the RoboMaster SDK network setup, and camera support through OpenCV.

## Experiments and Results

The experiment labels below match the sections in [robotarium_results.xlsx](robotarium/robotarium_results.xlsx).

| Experiment | Purpose | Main local scripts | Robotarium scripts | Result locations |
| --- | --- | --- | --- | --- |
| E1 | No-failure target search comparing centralized planning, STIGSAR/stigmergy search, and random walk baselines. | [simulation/centralized/centralized_parallel_experiment.py](simulation/centralized/centralized_parallel_experiment.py), [simulation/stigmergy_search/stigmergy_search_efficient_parallel_experiment.py](simulation/stigmergy_search/stigmergy_search_efficient_parallel_experiment.py), [simulation/random_walk/random_walk_parallel_experiment.py](simulation/random_walk/random_walk_parallel_experiment.py) | [robotarium/centralized_E1_E2/robotarium_centralized.py](robotarium/centralized_E1_E2/robotarium_centralized.py), [robotarium/stigmergy_search_E1_E2/robotarium_stigmergy_search.py](robotarium/stigmergy_search_E1_E2/robotarium_stigmergy_search.py), [robotarium/random_walk_E1_E2/robotarium_random_walk.py](robotarium/random_walk_E1_E2/robotarium_random_walk.py) | [simulation/centralized/experiment_results/E1](simulation/centralized/experiment_results/E1/), [simulation/stigmergy_search/experiment_results_stigmergy_efficient/E1](simulation/stigmergy_search/experiment_results_stigmergy_efficient/E1/), [simulation/random_walk/experiment_results_random_walk/E1](simulation/random_walk/experiment_results_random_walk/E1/), [robotarium_results.xlsx](robotarium/robotarium_results.xlsx) |
| E2 | Failure-robust target search using the same approaches with vehicle failures. | Same E1 scripts with failure counts enabled in the experiment configuration. | Same E1 Robotarium folders, using the E2 failure schedule/settings. | [simulation/centralized/experiment_results/E2](simulation/centralized/experiment_results/E2/), [simulation/stigmergy_search/experiment_results_stigmergy_efficient/E2](simulation/stigmergy_search/experiment_results_stigmergy_efficient/E2/), [simulation/random_walk/experiment_results_random_walk/E2](simulation/random_walk/experiment_results_random_walk/E2/), [robotarium_results.xlsx](robotarium/robotarium_results.xlsx) |
| E4 | Convergence/rendezvous behavior measured by target distance shells. | [simulation/stigmergy_convergence/rendezvous_parallel_experiment.py](simulation/stigmergy_convergence/rendezvous_parallel_experiment.py), [simulation/stigmergy_convergence/analyze_rendezvous_results.py](simulation/stigmergy_convergence/analyze_rendezvous_results.py) | [robotarium/convergence_E4/robotarium_rendezvous.py](robotarium/convergence_E4/robotarium_rendezvous.py) | [simulation/stigmergy_convergence/experiment_results_rendezvous](simulation/stigmergy_convergence/experiment_results_rendezvous/), [simulation/stigmergy_convergence/analysis_outputs_rendezvous](simulation/stigmergy_convergence/analysis_outputs_rendezvous/), [robotarium/convergence_video/stig_convergence.mp4](robotarium/convergence_video/stig_convergence.mp4), [robotarium_results.xlsx](robotarium/robotarium_results.xlsx) |
| E5 | Search-and-rescue and pheromone-ablation study comparing repulsive-only, attractive-only, dual pheromone, merged-field, and sign-flip variants. | [simulation/stigmergy_dual_behavior/stigmergy_search_and_rescue_parallel_experiment.py](simulation/stigmergy_dual_behavior/stigmergy_search_and_rescue_parallel_experiment.py), [simulation/stigmergy_dual_behavior/repulsive_only_parallel_experiment.py](simulation/stigmergy_dual_behavior/repulsive_only_parallel_experiment.py), [simulation/stigmergy_dual_behavior/attractive_only_parallel_experiment.py](simulation/stigmergy_dual_behavior/attractive_only_parallel_experiment.py), [simulation/stigmergy_dual_behavior/single_evaporative_merged_parallel_experiment.py](simulation/stigmergy_dual_behavior/single_evaporative_merged_parallel_experiment.py), [simulation/stigmergy_dual_behavior/single_persistent_merged_parallel_experiment.py](simulation/stigmergy_dual_behavior/single_persistent_merged_parallel_experiment.py), [simulation/stigmergy_dual_behavior/sign_flip_parallel_experiment.py](simulation/stigmergy_dual_behavior/sign_flip_parallel_experiment.py), [simulation/stigmergy_dual_behavior/analyze_e5_ablation_results.py](simulation/stigmergy_dual_behavior/analyze_e5_ablation_results.py) | [robotarium/dual_pheromone_search_and_rescue_E5/robotarium_dual_pheromone.py](robotarium/dual_pheromone_search_and_rescue_E5/robotarium_dual_pheromone.py), [robotarium/repulsive_E5/robotarium_single_pheromone.py](robotarium/repulsive_E5/robotarium_single_pheromone.py), [robotarium/attractive_E5/robotarium_single_pheromone.py](robotarium/attractive_E5/robotarium_single_pheromone.py), [robotarium/single_evap_pheromone_search_rescue_E5/robotarium_single_pheromone.py](robotarium/single_evap_pheromone_search_rescue_E5/robotarium_single_pheromone.py), [robotarium/single_persistent_pheromone_search_rescue_E5/robotarium_single_pheromone.py](robotarium/single_persistent_pheromone_search_rescue_E5/robotarium_single_pheromone.py), [robotarium/sign_flip_E5/robotarium_single_pheromone.py](robotarium/sign_flip_E5/robotarium_single_pheromone.py) | [simulation/stigmergy_dual_behavior/E5](simulation/stigmergy_dual_behavior/E5/), [simulation/stigmergy_dual_behavior/E5/analysis_outputs](simulation/stigmergy_dual_behavior/E5/analysis_outputs/), [robotarium_results.xlsx](robotarium/robotarium_results.xlsx) |

## Running Simulation Experiments

Run commands from the repository root unless a command changes directories. The parallel scripts accept `--workers N`; omit the option to use all available CPU cores.

The E1/E2 simulation scripts import a `config_experiments` module; use the same schema shown in [robotarium/stigmergy_search_E1_E2/config_experiments.py](robotarium/stigmergy_search_E1_E2/config_experiments.py), setting `FAILURE_COUNTS` to all zeros for E1 and to positive failure counts for E2.

```bash
cd simulation/centralized
python centralized_parallel_experiment.py --workers 4

cd ../stigmergy_search
python stigmergy_search_efficient_parallel_experiment.py --workers 4

cd ../random_walk
python random_walk_parallel_experiment.py --workers 4
```

For E4:

```bash
cd simulation/stigmergy_convergence
python rendezvous_parallel_experiment.py --workers 4
python analyze_rendezvous_results.py
```

For E5:

```bash
cd simulation/stigmergy_dual_behavior
python stigmergy_search_and_rescue_parallel_experiment.py --workers 4
python repulsive_only_parallel_experiment.py --workers 4
python attractive_only_parallel_experiment.py --workers 4
python single_evaporative_merged_parallel_experiment.py --workers 4
python single_persistent_merged_parallel_experiment.py --workers 4
python sign_flip_parallel_experiment.py --workers 4
python analyze_e5_ablation_results.py
```

## Running Robotarium Experiments

The [Robotarium](https://www.robotarium.gatech.edu/) is Georgia Tech's remote-access multi-robot research facility; for research use, cite the Robotarium references requested on the official Robotarium site, especially Wilson et al. 2020 for experiments after January 2019.

```bash
cd robotarium/centralized_E1_E2
python robotarium_centralized.py --no-show

cd ../stigmergy_search_E1_E2
python robotarium_stigmergy_search.py --no-show

cd ../random_walk_E1_E2
python robotarium_random_walk.py --no-show

cd ../convergence_E4
python robotarium_rendezvous.py --no-show

cd ../dual_pheromone_search_and_rescue_E5
python robotarium_dual_pheromone.py --no-show
```

Run the other E5 Robotarium variants from their matching folders: [repulsive_E5](robotarium/repulsive_E5/), [attractive_E5](robotarium/attractive_E5/), [single_evap_pheromone_search_rescue_E5](robotarium/single_evap_pheromone_search_rescue_E5/), [single_persistent_pheromone_search_rescue_E5](robotarium/single_persistent_pheromone_search_rescue_E5/), and [sign_flip_E5](robotarium/sign_flip_E5/).

The E4 Robotarium convergence demonstration video is shown below and is also available at [robotarium/convergence_video/stig_convergence.mp4](robotarium/convergence_video/stig_convergence.mp4).

<video controls width="720">
  <source src="robotarium/convergence_video/stig_convergence.mp4" type="video/mp4">
  Your browser does not support embedded MP4 playback.
</video>

## Running RoboMaster Experiments

Physical RoboMaster experiments are run from [robomaster/scripts](robomaster/scripts/) after connecting the robots and confirming the serial numbers, arena dimensions, and camera/SDK setup in each script.

```bash
cd robomaster/scripts
python centralized.py
python centralized_with_failure.py
python random_walk.py
python random_walk_failure.py
python stigmergy_search.py
python stigmergy_search_failure.py
```

Local metadata summaries are in [robomaster/centralized_results](robomaster/centralized_results/), [robomaster/random_walk_results](robomaster/random_walk_results/), [robomaster/stigmergy_results](robomaster/stigmergy_results/), [coverage_analysis_E1.csv](robomaster/coverage_analysis_E1.csv), and [coverage_analysis_E2.csv](robomaster/coverage_analysis_E2.csv).

## Git LFS and Commit Workflow

The largest files are generated result artifacts, especially E5 path/convergence CSVs, time-series CSVs, Excel workbooks, NumPy arrays, and RoboMaster run metadata. Track these with Git LFS before adding the current repository state:

```bash
git lfs install
git lfs track "*.xlsx"
git lfs track "*.npy"
git lfs track "simulation/**/*.csv"
git lfs track "robomaster/**/*metadata.json"
git add .gitattributes
git add -A
git status
git lfs ls-files
git commit -m "Document experiment layout and results"
```

If any of these result files were committed before LFS tracking was enabled, migrate them instead of only tracking future files:

```bash
git lfs migrate import --include="*.xlsx,*.npy,simulation/**/*.csv,robomaster/**/*metadata.json"
```

## References

- [Robotarium, Georgia Institute of Technology](https://www.robotarium.gatech.edu/)
- [Robotarium Python Simulator](https://github.com/robotarium/robotarium_python_simulator)
- Sean Wilson et al., "The Robotarium: Globally Impactful Opportunities, Challenges, and Lessons Learned in Remote-Access, Distributed Control of Multirobot Systems," IEEE Control Systems Magazine, 2020.
- Daniel Pickem et al., "The Robotarium: A remotely accessible swarm robotics research testbed," IEEE International Conference on Robotics and Automation, 2017.
