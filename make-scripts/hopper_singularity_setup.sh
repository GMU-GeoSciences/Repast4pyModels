#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "============================="
echo "== Setting up Slurm env... =="
echo "============================="

echo "Loading modules..."
module load gnu10 openmpi  git singularity

echo "Building Singularity Container..."
cd /containers/hopper/UserContainers/$USER
singularity build repast4py_latest.sif docker:ghcr.io/gmu-geosciences/repast4py-container:latest

echo "============================="
echo "== Environment ready to go =="
echo "============================="

echo "Setting up Slurm..."
salloc -p normal -q normal -n 1 --ntasks-per-node=24 --mem=50GB
