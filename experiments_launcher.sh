#!/bin/sh

## train on indomain_train and finetune on oodomain_train
# variant -> baseline
./run_experiments.sh train baseline default
./run_experiments.sh finetune baseline default-finetune save/baseline.default-01/checkpoint
# variant -> baseline-cond
./run_experiments.sh train baseline-cond default
./run_experiments.sh finetune baseline-cond default-finetune save/baseline-cond.default-01/checkpoint
# variant -> baseline-cond-att
./run_experiments.sh train baseline-cond-att default
./run_experiments.sh finetune baseline-cond-att default-finetune save/baseline-cond-att.default-01/checkpoint
# variant -> qagan
./run_experiments.sh train qagan default
./run_experiments.sh finetune qagan default-finetune save/qagan.default-01/checkpoint
# variant -> qagan-hidden
./run_experiments.sh train qagan-hidden default
./run_experiments.sh finetune qagan-hidden default-finetune save/qagan-hidden.default-01/checkpoint
# variant -> qagan-cond
./run_experiments.sh train qagan-cond default
./run_experiments.sh finetune qagan-cond default-finetune save/qagan-cond.default-01/checkpoint
# variant -> qagan-cond-att
./run_experiments.sh train qagan-cond-att default
./run_experiments.sh finetune qagan-cond-att default-finetune save/qagan-cond-att.default-01/checkpoint

## evaluate on oodomain_val
# variant -> baseline
./run_experiments.sh evaluate baseline default
./run_experiments.sh evaluate baseline default-finetune
# variant -> baseline-cond
./run_experiments.sh evaluate baseline-cond default
./run_experiments.sh evaluate baseline-cond default-finetune
# variant -> baseline-cond-att
./run_experiments.sh evaluate baseline-cond-att default
./run_experiments.sh evaluate baseline-cond-att default-finetune
# variant -> qagan
./run_experiments.sh evaluate qagan default
./run_experiments.sh evaluate qagan default-finetune
# variant -> qagan-hidden
./run_experiments.sh evaluate qagan-hidden default
./run_experiments.sh evaluate qagan-hidden default-finetune
# variant -> qagan-cond
./run_experiments.sh evaluate qagan-cond default
./run_experiments.sh evaluate qagan-cond default-finetune
# variant -> qagan-cond-att
./run_experiments.sh evaluate qagan-cond-att default
./run_experiments.sh evaluate qagan-cond-att default-finetune