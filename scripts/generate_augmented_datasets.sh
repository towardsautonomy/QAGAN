#!/bin/sh

python src/data_augmentation.py --domain="indomain_train" --dataset=squad
python src/data_augmentation.py --domain="indomain_train" --dataset=newsqa
python src/data_augmentation.py --domain="indomain_train" --dataset=nat_questions

python src/data_augmentation.py --domain="oodomain_train" --dataset=duorc
python src/data_augmentation.py --domain="oodomain_train" --dataset=race
python src/data_augmentation.py --domain="oodomain_train" --dataset=relation_extraction