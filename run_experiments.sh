#!/bin/bash

# color codes
black=\\e[0m
cyan=\\e[96m
red=\\e[91m
reset=\\033[0m

# function to print usage
print_usage() {
    echo -e "    -> Usage: ./run_experiments.sh                     \\
                        {train|finetune|evaluate|test} \\
                        {variant}                      \\
                        {experiment_name}              \\
                        {pretrained_checkpoint}${reset}"
}

# experiment mode
MODE=$1
variant=$2
experiment=$3
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo -e "${red}Incorrect arguments."
    print_usage
    exit 1
elif [ "$#" -eq 4 ]; then
    if [ "$MODE" != "finetune" ]; then
        echo ${MODE}
        echo -e "${red}Pretrained model should only be used during finetuning."
        print_usage
        exit 1
    fi
    pretrained_ckpt=$4
fi

## run experiments
if [ "$MODE" == "train" ]; then
    # train
    python run.py --do-train \
                  --variant ${variant} \
                  --eval-every 2000 --run-name ${experiment} --recompute-features

elif [ "$MODE" == "finetune" ]; then
    # train
    python run.py --do-train \
                  --variant ${variant} \
                  --eval-every 10 --num-epochs 10 --run-name ${experiment} \
                  --finetune --pretrained-model=${pretrained_ckpt} --recompute-features

elif [ "$MODE" == "evaluate" ]; then
    # evaluate
   python run.py --do-eval \
                 --variant ${variant} \
                 --run-name ${experiment} \
                 --sub-file mtl_submission_val.csv \
                 --save-dir save/${variant}.${experiment}-01 --eval-dir datasets/oodomain_val --recompute-features

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