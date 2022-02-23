#!/bin/sh

# train
./run_experiments.sh train baseline-v0 init
./run_experiments.sh train baseline-v1 init
./run_experiments.sh train baseline-v3 init
./run_experiments.sh train baseline-v4 init
./run_experiments.sh train qagan-v0 init
./run_experiments.sh train qagan-v1 init
./run_experiments.sh train qagan-v2 init

# evaluate
./run_experiments.sh evaluate baseline-v0 init
./run_experiments.sh evaluate baseline-v1 init
./run_experiments.sh evaluate baseline-v3 init
./run_experiments.sh evaluate baseline-v4 init
./run_experiments.sh evaluate qagan-v0 init
./run_experiments.sh evaluate qagan-v1 init
./run_experiments.sh evaluate qagan-v2 init