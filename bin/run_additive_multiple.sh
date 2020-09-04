#!/bin/bash

date=$1

declare -a StringArray=("heart" "liver" "pima")

for dataset in "${StringArray[@]}";do
	echo $dataset
	python3 ../experimental/experiment_additive.py $date $dataset
done