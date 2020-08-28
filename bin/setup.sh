#!/bin/bash

echo "Create a separate environment!"

ENV="gp"

conda create -y -n $ENV --clone root

source activate $ENV

pip install --upgrade tensorflow==2.2.0
pip install --upgrade gpflow==2.0.5
pip install --upgrade tensorflow_datasets==3.2.1