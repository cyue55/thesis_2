#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <recipe_name>"
    exit 1
fi
mkdir -p "recipes/${1}/models/example_model"
ln -s ../../data "recipes/${1}/data"
ln -s ../../lsf "recipes/${1}/lsf"
ln -s ../../scripts "recipes/${1}/scripts"
ln -s ../../venv "recipes/${1}/venv"
ln -s scripts/train.py "recipes/${1}/train.py"
cp recipes/example/models/example_model/config.yaml "recipes/${1}/models/example_model/config.yaml"
