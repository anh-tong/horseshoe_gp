#!/bin/bash

date=$1

declare -a StringArray=("housing" "concrete" "energy" "kin8nm" "naval" "power_plant" "wine" "yatch")

for dataset in "${StringArray[@]}";do
	echo $dataset
	python3 ../experimental/experiment_uci_no_prior.py $date $dataset
done
