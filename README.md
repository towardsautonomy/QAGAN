# QAGAN: Adversarial Approach To Learning Domain Invariant Language Features

## Getting Started
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `cd environment && conda env create -f conda_env.yml`
- Usage: `./run_experiments.sh {MODE} {VARIANT} {EXPERIMENT_NAME}`
- e.g:
    - Train a baseline MTL system with `./run_experiments.sh train baseline baseline`
    - Evaluate the system on test set with `./run_experiments.sh test baseline baseline`
    - Upload the csv file in `save/baseline-01` to the test leaderboard. For the validation leaderboard, run `./run_experiments.sh evaluate baseline baseline`

# Experiments

| Variant     | Split           | F1          | EM          |  
| ----------- | --------------- | ----------- | ----------- |
| Baseline    | indomain_val    | 70.49       | 54.48       |
| Baseline    | oodomain_val    | 48.29       | 30.89       |