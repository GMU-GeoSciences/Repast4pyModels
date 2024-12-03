#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "============================="
echo "== Setting up Slurm env... =="
echo "============================="

echo "Loading modules..."
module load gnu10
module load openmpi
# module load git

echo "Creating conda environment..."
source miniconda/bin/activate 
conda create --name repast4py_env python=3.9 pip
conda activate repast4py_env
env CC=mpicxx CXX=mpicxx pip install repast4py typing-extensions packaging snuggs pandas

# echo "Cloning project from git..."
# git clone https://github.com/heykuldip/Repast4pyModels.git

echo "Setting up Slurm..."
salloc -p normal -q normal -n 1 --ntasks-per-node=24 --mem=50GB

echo "============================="
echo "== Environment ready to go =="
echo "============================="
