# QAGAN: Adversarial Approach To Learning Domain Invariant Language Features

## Getting Started
- Download datasets from [here](https://drive.google.com/file/d/1Fv2d30hY-2niU7t61ktnMsi_HUXS6-Qx/view?usp=sharing)
- Setup environment with `cd environment && conda env create -f conda_env.yml`
- Usage: 
    ```/bin/sh
    Usage: ./run_experiments.sh train {variant} {experiment_name}
           ./run_experiments.sh finetune {variant} {experiment_name} {pretrained_ckpt_path}
           ./run_experiments.sh evaluate {variant} {experiment_name}
           ./run_experiments.sh evaluate {test} {experiment_name}
    ```
- e.g:
    - Train a baseline MTL system with `./run_experiments.sh train qagan default`
    - Train a baseline MTL system with `./run_experiments.sh finetune qagan-finetune default save/qagan.default-01`
    - Evaluate the system on validation set, run `./run_experiments.sh evaluate qagan-finetune default`
    - Evaluate the system on test set with `./run_experiments.sh test qagan-finetune default`
- For submitting to leaderboard, upload the csv file generated in `save/{VARIANT}.{EXPERIMENT_NAME}-01` to the test leaderboard.
- To run all the experiments, run `sh experiments_launcher.sh`.  

## Quantitative Evaluation

![](media/quantitative_evaluation.png)
