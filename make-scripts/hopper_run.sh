#!/bin/bash
# Script to run the repast script in hopper

echo "Running repast4py script on Hopper..."
cd ./repast4py
mpirun -n 4 python rndwalk.py random_walk.yaml
