#!/bin/sh

# train
./run_experiments.sh train qagan-v0 disc-warmup-anneal
# ./run_experiments.sh train qagan-v1 disc-warmup-anneal
# ./run_experiments.sh train qagan-v2 disc-warmup-anneal

# evaluate
./run_experiments.sh evaluate qagan-v0 disc-warmup-anneal
# ./run_experiments.sh evaluate qagan-v1 disc-warmup-anneal
# ./run_experiments.sh evaluate qagan-v2 disc-warmup-anneal