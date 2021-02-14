#!/usr/bin/env bash

DATASET=MAG
MODEL=MATCH

PYTHONFAULTHANDLER=1 python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --mode train --reg 1
PYTHONFAULTHANDLER=1 python main.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --mode eval

python evaluation.py \
--results $DATASET/results/$MODEL-$DATASET-labels.npy \
--targets $DATASET/test_labels.npy \
--train-labels $DATASET/train_labels.npy