#!/bin/bash

# color codes
black=\\e[0m
cyan=\\e[96m
red=\\e[91m
reset=\\033[0m

# function to print usage
print_usage() {
    echo -e "    -> Usage: ./run_experiments.sh train|evaluate|test experiment_name${reset}"
}

# experiment mode
MODE=$1
experiment=$2
if [ "$#" -ne 2 ]; then
    echo -e "${red}Please provide experiment mode and experiment name."
    print_usage
    exit 1
fi

## run experiments
if [ "$MODE" == "train" ]; then
    # train
    python run.py --do-train \
                  --eval-every 2000 --run-name "$experiment"

elif [ "$MODE" == "evaluate" ]; then
    # evaluate
   python run.py --do-eval \
                 --sub-file mtl_submission_val.csv \
                 --save-dir save/"$experiment"-01 --eval-dir datasets/oodomain_val

elif [ "$MODE" == "test" ]; then
	# evaluate
    python train.py --do-eval 
                    --sub-file mtl_submission.csv 
                    --save-dir save/"$experiment"-01
else
    # print error
    echo -e "${red}Error:${reset} Invalid mode"
    print_usage
fi