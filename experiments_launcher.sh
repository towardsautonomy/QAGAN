#!/bin/sh

# train
./run_experiments.sh train baseline-v0 default
./run_experiments.sh train baseline-v1 default
./run_experiments.sh train baseline-v3 default
./run_experiments.sh train baseline-v4 default
./run_experiments.sh train qagan-v0 default
./run_experiments.sh train qagan-v1 default
./run_experiments.sh train qagan-v2 default

# evaluate
./run_experiments.sh evaluate baseline-v0 default
./run_experiments.sh evaluate baseline-v1 default
./run_experiments.sh evaluate baseline-v3 default
./run_experiments.sh evaluate baseline-v4 default
./run_experiments.sh evaluate qagan-v0 default
./run_experiments.sh evaluate qagan-v1 default
./run_experiments.sh evaluate qagan-v2 default