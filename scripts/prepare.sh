#!/bin/bash

input=../data/inputs/train_data.csv
input_test=../data/inputs/test_sequences.csv
output=../data/model_inputs

python prepare_base.py $input $input_test $output

ls -lh $output



