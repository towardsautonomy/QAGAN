#!/bin/bash

# color codes
black=\\e[0m
cyan=\\e[96m
red=\\e[91m
reset=\\033[0m

# function to print usage
print_usage() {
    echo -e "Usage: $0 train {variant} {experiment_name}"
    echo -e "       $0 finetune {variant} {experiment_name} {base_model} {pretrained_ckpt_path}"
    echo -e "       $0 evaluate {variant} {experiment_name} {base_model} {dataset}"
    echo -e "       $0 test {variant} {experiment_name} {base_model} {dataset}${reset}"
}

# experiment mode
MODE=$1
variant=$2
experiment=$3
base_model=$4
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo -e "${red}Incorrect arguments."
    print_usage
    exit 1
elif [ "$#" -eq 5 ]; then
    if [ "$MODE" = "finetune" ]; then
        pretrained_ckpt=$5
    else
        dataset=$5
    fi
    
    
fi

## run experiments
if [ "$MODE" == "train" ]; then
    # train
    python run.py --do-train \
                  --variant ${variant} \
                  --eval-every 2000 --run-name ${experiment} #--recompute-features

elif [ "$MODE" == "finetune" ]; then
    # train
    python run.py --do-train \
                  --variant ${variant} \
                  --eval-every 10 \
                  --num-epochs 10 \
                  --run-name ${experiment} \
                  --base-model=${base_model} \
                  --finetune --pretrained-model=${pretrained_ckpt} #--recompute-features

elif [ "$MODE" == "evaluate" ]; then
    # evaluate
   python run.py --do-eval \
                 --variant ${variant} \
                 --run-name ${experiment} \
                 --sub-file mtl_submission_val.csv \
                 --output-dir output/${variant}.${base_model}.${experiment}-01 \
                 --eval-dir datasets/oodomain_val \
                 --base-model=${base_model} \
                 --eval-datasets=${dataset} #--recompute-features

elif [ "$MODE" == "test" ]; then
	# evaluate
    python run.py --do-eval \
                  --variant ${variant} \
                  --run-name ${experiment} \
                  --sub-file mtl_submission.csv \
                  --base-model=${base_model} \
                  --output-dir output/${variant}.${base_model}.${experiment}-01 \
                  --eval-datasets=${dataset}
else
    # print error
    echo -e "${red} Invalid mode"
    print_usage
fi