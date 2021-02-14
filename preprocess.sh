#!/usr/bin/env bash

DATASET=MAG

python transform_data.py --dataset $DATASET

python preprocess.py \
--text-path $DATASET/train_texts.txt \
--label-path $DATASET/train_labels.txt \
--vocab-path $DATASET/vocab.npy \
--emb-path $DATASET/emb_init.npy \
--w2v-model $DATASET/$DATASET.joint.emb \

python preprocess.py \
--text-path $DATASET/test_texts.txt \
--label-path $DATASET/test_labels.txt \
--vocab-path $DATASET/vocab.npy \
