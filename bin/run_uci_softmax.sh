#!/bin/bash


declare -a StringArray=("housing" "concrete" "energy" "kin8nm" "naval" "power_plant" "wine" "yatch")
declare -a id=("0910_1_softmax" "0910_2_softmax" "0910_3_sotfmax")

for date in "${id[@]}";do
  echo $date
  for dataset in "${StringArray[@]}";do
    echo $dataset
    python3 ../experimental/experiment_uci_softmax.py $date $dataset
  done
done
