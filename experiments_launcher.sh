#!/bin/sh

## train on indomain_train and finetune on oodomain_train
# variant -> baseline
./run_experiments.sh train baseline mrqa
# variant -> baseline-cond
./run_experiments.sh train baseline-cond mrqa
# variant -> baseline-cond-att
./run_experiments.sh train baseline-cond-att mrqa
# variant -> qagan
./run_experiments.sh train qagan mrqa
# variant -> qagan-hidden
./run_experiments.sh train qagan-hidden mrqa
# variant -> qagan-cond
./run_experiments.sh train qagan-cond mrqa
# variant -> qagan-cond-att
./run_experiments.sh train qagan-cond-att mrqa

## evaluate on oodomain_val
# variant -> baseline
./run_experiments.sh evaluate baseline mrqa distil-bert BioASQ
./run_experiments.sh evaluate baseline mrqa distil-bert DROP
./run_experiments.sh evaluate baseline mrqa distil-bert DuoRC
./run_experiments.sh evaluate baseline mrqa distil-bert RACE
./run_experiments.sh evaluate baseline mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate baseline mrqa distil-bert TextbookQA
# variant -> baseline-cond
./run_experiments.sh evaluate baseline-cond mrqa distil-bert BioASQ
./run_experiments.sh evaluate baseline-cond mrqa distil-bert DROP
./run_experiments.sh evaluate baseline-cond mrqa distil-bert DuoRC
./run_experiments.sh evaluate baseline-cond mrqa distil-bert RACE
./run_experiments.sh evaluate baseline-cond mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate baseline-cond mrqa distil-bert TextbookQA
# variant -> baseline-cond-att
./run_experiments.sh evaluate baseline-cond-att mrqa distil-bert BioASQ
./run_experiments.sh evaluate baseline-cond-att mrqa distil-bert DROP
./run_experiments.sh evaluate baseline-cond-att mrqa distil-bert DuoRC
./run_experiments.sh evaluate baseline-cond-att mrqa distil-bert RACE
./run_experiments.sh evaluate baseline-cond-att mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate baseline-cond-att mrqa distil-bert TextbookQA
# variant -> qagan
./run_experiments.sh evaluate qagan mrqa distil-bert BioASQ
./run_experiments.sh evaluate qagan mrqa distil-bert DROP
./run_experiments.sh evaluate qagan mrqa distil-bert DuoRC
./run_experiments.sh evaluate qagan mrqa distil-bert RACE
./run_experiments.sh evaluate qagan mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate qagan mrqa distil-bert TextbookQA
# variant -> qagan-hidden
./run_experiments.sh evaluate qagan-hidden mrqa distil-bert BioASQ
./run_experiments.sh evaluate qagan-hidden mrqa distil-bert DROP
./run_experiments.sh evaluate qagan-hidden mrqa distil-bert DuoRC
./run_experiments.sh evaluate qagan-hidden mrqa distil-bert RACE
./run_experiments.sh evaluate qagan-hidden mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate qagan-hidden mrqa distil-bert TextbookQA
# variant -> qagan-cond
./run_experiments.sh evaluate qagan-cond mrqa distil-bert BioASQ
./run_experiments.sh evaluate qagan-cond mrqa distil-bert DROP
./run_experiments.sh evaluate qagan-cond mrqa distil-bert DuoRC
./run_experiments.sh evaluate qagan-cond mrqa distil-bert RACE
./run_experiments.sh evaluate qagan-cond mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate qagan-cond mrqa distil-bert TextbookQA
# variant -> qagan-cond-att
./run_experiments.sh evaluate qagan-cond-att mrqa distil-bert BioASQ
./run_experiments.sh evaluate qagan-cond-att mrqa distil-bert DROP
./run_experiments.sh evaluate qagan-cond-att mrqa distil-bert DuoRC
./run_experiments.sh evaluate qagan-cond-att mrqa distil-bert RACE
./run_experiments.sh evaluate qagan-cond-att mrqa distil-bert RelationExtraction
./run_experiments.sh evaluate qagan-cond-att mrqa distil-bert TextbookQA