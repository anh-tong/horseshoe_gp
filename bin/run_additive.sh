#!/bin/bash

date=$1

declare -a StringArray=("concrete" "energy" "kin8nm" "naval" "power_plant" "wine" "yatch")

for dataset in "${StringArray[@]}";do
	echo $dataset
	python3 ../experimental/experiment_uci.py $date $dataset
done