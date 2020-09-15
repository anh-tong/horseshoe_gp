#!/bin/bash

declare -a dates=("additive_no_prior_1"
                  "additive_no_prior_2"
                  "additive_no_prior_3"
                   "additive_no_prior_4"
                   "additive_no_prior_5"
                   "additive_no_prior_6"
                   "additive_no_prior_7"
                   "additive_no_prior_8"
                   "additive_no_prior_9"
                   "additive_no_prior_10")
declare -a StringArray=("heart" "liver" "pima")

for date in "${dates[@]}"; do
  for dataset in "${StringArray[@]}";do
    echo $dataset
    python3 ../experimental/experiment_additive_no_prior.py $date $dataset
  done
done