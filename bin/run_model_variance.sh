#!/bin/bash

declare -a choices=("naive" "ours")

for choice in "${choices[@]}"; do
  for id in {1..10};do
    echo $dataset
    python3 ../experimental/experiment_model_variance.py $choice $id
  done
done