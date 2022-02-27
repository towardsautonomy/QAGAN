# QAGAN: Adversarial Approach To Learning Domain Invariant Language Features

## Getting Started
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `cd environment && conda env create -f conda_env.yml`
- Usage: `./run_experiments.sh {MODE {VARIANT {EXPERIMENT_NAME}`
- e.g:
    - Train a baseline MTL system with `./run_experiments.sh train baseline-v0 default`
    - Evaluate the system on test set with `./run_experiments.sh test baseline-v0 default`
    - Evaluate the system on validation set, run `./run_experiments.sh evaluate baseline-v0 default`
- For submitting to leaderboard, upload the csv file generated in `save/{VARIANT}.{EXPERIMENT_NAME}-01` to the test leaderboard.
- To run all the experiments, run `sh experiments_launcher.sh`.  

# Experiments

| **Method**            | **indomain_val** | **oodomain_val** | **oodomain_test** |       |    |    |
|-----------------------|------------------|------------------|-------------------|-------|----|----|
|                       | F1               | EM               | F1                | EM    | F1 | EM |
| baseline              | 70.49            | 54.48            | 48.29             | 30.89 | -  | -  |
| qagan-kld             | 70.10            | 54.24            | 46.56             | 31.15 | -  | -  |
| qagan-kld-finetune    | -                | -                | 47.38             | 33.25 | -  | -  |
| qagan-hidden          | 68.88            | 52.69            | 46.95             | 30.89 | -  | -  |
| qagan-hidden-finetune | -                | -                | 48.46             | 34.03 | -  | -  |
| qagan                 | 69.85            | 53.84            | 46.92             | 31.68 | -  | -  |
| qagan-finetune        | -                | -                | 49.16             | 34.03 | -  | -  |
| qagan-cond            | 70.00            | 53.84            | 49.38             | 34.29 | -  | -  |
| qagan-cond-finetune   | -                | -                | 51.00             | 35.60 | -  | -  |
