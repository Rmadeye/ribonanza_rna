#!/bin/bash


input=../data/inputs/train_data.csv
input_test=../data/inputs/test_sequences.csv
output=../data/model_inputs

python prepare_base.py $output --sn 0.6 # for example purposes, we are using a small number of sequence




