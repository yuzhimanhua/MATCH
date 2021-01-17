#!/usr/bin/env bash

DATA_PATH=data
DATASET=MeSH
MODEL=MATCH

PYTHONFAULTHANDLER=1 python main_mesh.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --mode train
PYTHONFAULTHANDLER=1 python main_mesh.py --data-cnf configure/datasets/$DATASET.yaml --model-cnf configure/models/$MODEL-$DATASET.yaml --mode eval

python evaluation.py \
--results $DATA_PATH/$DATASET/results/$MODEL-$DATASET-labels.npy \
--targets $DATA_PATH/$DATASET/test_labels.npy \
--train-labels $DATA_PATH/$DATASET/train_labels.npy