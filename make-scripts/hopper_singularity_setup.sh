#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "============================="
echo "== Setting up Slurm env... =="
echo "============================="

echo "Loading modules..." 
module load gnu9 intel mpich git singularity

echo "Current Singularity containers:"
ls -lah /containers/hopper/UserContainers/$USER/

echo "Building Singularity Container..."
singularity build /containers/hopper/UserContainers/$USER/repast4py_latest.sif docker:ghcr.io/gmu-geosciences/repast4py-container:latest
# cp repast4py_latest.sif /containers/hopper/UserContainers/$USER/.

echo "============================="
echo "== Environment ready to go =="
echo "============================="
