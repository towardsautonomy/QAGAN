#!/bin/bash

# color codes
black=\\e[0m
cyan=\\e[96m
red=\\e[91m
reset=\\033[0m

# function to print usage
print_usage() {
    echo -e "    -> Usage: ./run_experiments.sh [train|evaluate|test] {variant} [experiment_name]${reset}"
}

# experiment mode
MODE=$1
variant=$2
experiment=$3
if [ "$#" -ne 3 ]; then
    echo -e "${red}Incorrect arguments."
    print_usage
    exit 1
fi

## run experiments
if [ "$MODE" == "train" ]; then
    # train
    python run.py --do-train \
                  --variant ${variant} \
                  --eval-every 2000 --run-name ${experiment} #--recompute-features

elif [ "$MODE" == "evaluate" ]; then
    # evaluate
   python run.py --do-eval \
                 --variant ${variant} \
                 --run-name ${experiment} \
                 --sub-file mtl_submission_val.csv \
                 --save-dir save/${variant}.${experiment}-01 --eval-dir datasets/oodomain_val

elif [ "$MODE" == "test" ]; then
	# evaluate
    python run.py --do-eval \
                  --variant ${variant} \
                  --run-name ${experiment} \
                  --sub-file mtl_submission.csv \
                  --save-dir save/${variant}.${experiment}-01
else
    # print error
    echo -e "${red} Invalid mode"
    print_usage
fi