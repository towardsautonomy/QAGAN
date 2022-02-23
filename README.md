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

## Definition of Various Variants
| Variant     | Prediction Head    | Discriminator Input | Discriminator | Logits Conditioned | Embedding Dist Reg | Include OOD in train |
| ----------- | ------------------ | ------------------- | ------------- | ------------------ |------------------- | -------------------- |
| baseline-v0 | default            | ☐                   | ✗             | ✗                  | ✗                  |  ✗                   |
| baseline-v1 | 2 layer MLP        | ☐                   | ✗             | ✗                  | ✗                  |  ✗                   |
| baseline-v3 | MLP Conditional    | ☐                   | ✗             | ✗                  | ✗                  |  ✗                   |
| baseline-v4 | Self-Attention Conditional | ☐           | ✗             | ✗                  | ✗                  |  ✗                   |
| qagan-v0    | 2 layer MLP        | \<CLS\>             | ✓             | ✗                  | ✗                  |  ✗                   |
| qagan-v1    | 2 layer MLP        | [\<CLS\>\<SEP\>]    | ✓             | ✗                  | ✗                  |  ✗                   |
| qagan-v2    | 2 layer MLP        | hidden layers       | ✓             | ✗                  | ✗                  |  ✗                   |

## Experiment Results (DistilBERT)
| Variant     | Split           | Discriminator Anneal | F1          | EM          |  
| ----------- | --------------- | -------------------- | ----------- | ----------- |
| baseline-v0 | indomain_val    | ☐                    | 70.49       | 54.48       |
| baseline-v0 | oodomain_val    | ☐                    | 48.29       | 30.89       |
| baseline-v1 | indomain_val    | ☐                    | 70.65       | 54.67       |
| baseline-v1 | oodomain_val    | ☐                    | 48.53       | 34.03       |
| Baseline-v2 | indomain_val    | ☐                    | 70.72       | 54.84       |
| Baseline-v2 | oodomain_val    | ☐                    | 50.11       | 34.82       |
| baseline-v3 | indomain_val    | ☐                    | 69.34       | 53.57       |
| baseline-v3 | oodomain_val    | ☐                    | 45.40       | 31.68       |
| baseline-v4 | indomain_val    | ☐                    | 70.19       | 53.98       |
| baseline-v4 | oodomain_val    | ☐                    | 48.00       | 32.20       |
| qagan-v0    | indomain_val    | ✗                    | 70.20       | 54.24       |
| qagan-v0    | oodomain_val    | ✗                    | 50.65       | 34.03       |
| qagan-v0    | indomain_val    | ✓                    | 70.45       | 54.26       |
| qagan-v0    | oodomain_val    | ✓                    | 49.08       | 32.46       |
| qagan-v1    | oodomain_val    | ✗                    | 46.97       | 31.41       |
| qagan-v1    | indomain_val    | ✓                    | 70.35       | 54.00       |
| qagan-v1    | oodomain_val    | ✓                    | 47.84       | 34.29       |
| qagan-v2    | indomain_val    | ✓                    | 70.51       | 54.31       |
| qagan-v2    | oodomain_val    | ✓                    | 48.57       | 34.29       |
