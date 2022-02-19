# QAGAN: Adversarial Approach To Learning Domain Invariant Language Features

## Getting Started
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `cd environment && conda env create -f conda_env.yml`
- Usage: `./run_experiments.sh {MODE} {VARIANT} {EXPERIMENT_NAME}`
- e.g:
    - Train a baseline MTL system with `./run_experiments.sh train baseline-v0 default`
    - Evaluate the system on test set with `./run_experiments.sh test baseline-v0 default`
    - Evaluate the system on validation set, run `./run_experiments.sh evaluate baseline-v0 default`
- For submitting to leaderboard, upload the csv file generated in `save/{VARIANT}.{EXPERIMENT_NAME}-01` to the test leaderboard.

# Experiments

## Definition of various variants
| Variant     | Prediction Head | Discriminator | Attention-Based Pred Head | Conditional Pred Head | Embedding Dist Reg |
| ----------- | --------------- | ------------- | ------------------------- | --------------------- | ------------------ |
| baseline-v0 | default         | ✗             | ✗                         | ✗                     |  ✗                 |
| baseline-v1 | 2 layer MLP     | ✗             | ✗                         | ✗                     |  ✗                 |
| qagan-v0    | 2 layer MLP     | ✓             | ✗                         | ✗                     |  ✗                 |

## Experiment Results
| Variant     | Split           | F1          | EM          |  
| ----------- | --------------- | ----------- | ----------- |
| Baseline-v0 | indomain_val    | 70.49       | 54.48       |
| Baseline-v0 | oodomain_val    | 48.29       | 30.89       |
| Baseline-v1 | indomain_val    | 70.65       | 54.67       |
| Baseline-v1 | oodomain_val    | 48.53       | 34.03       |
