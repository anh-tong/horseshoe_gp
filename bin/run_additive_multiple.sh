#!/bin/bash

declare -a dates=("0903_4" "0903_5" "0903_6" "0903_7" "0903_8" "0903_9" "0903_10")
declare -a StringArray=("heart" "liver" "pima")

for date in "${dates[@]}"; do
  for dataset in "${StringArray[@]}";do
    echo $dataset
    python3 ../experimental/experiment_additive.py $date $dataset
  done
done