#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "============================="
echo "== Setting up Slurm env... =="
echo "============================="

echo "Loading modules..."
module load singularity
module load openmpi

echo "Building Singularity Container..."
cd /containers/hopper/UserContainers/$USER

singularity build repast4py_latest.sif docker:ghcr.io/gmu-geosciences/repast4py-container:latest

echo "Cloning project from git..."
cd ~
git clone https://github.com/heykuldip/Repast4pyModels.git

echo "============================="
echo "== Environment ready to go =="
echo "============================="