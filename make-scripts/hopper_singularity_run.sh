#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "=============================="
echo "== Running Repast Script... =="
echo "==============================" 

echo "Setting up Slurm..."
salloc -p normal -q normal -n 1 --ntasks-per-node=24 --mem=50GB

echo "Running repast4py script on Hopper..."
cd ./repast4py 

singularity run --nv /containers/hopper/UserContainers/$USER/repast4py_latest.sif mpirun -n 4 python rndwalk.py random_walk.yaml