#!/bin/bash
# This is the WRONG way to do it. This should be rin by a slurm script.
# This currently runs on the head node, which is bad. In interactive mode
# it should be run by a salloc kinda thing...

echo "=============================="
echo "== Running Repast Model...  =="
echo "==============================" 
echo "Running Singularity container..."
singularity run /containers/hopper/UserContainers/$USER/repast4py_latest.sif mpirun -n 4 python ./repast4py/deer_model.py ./config/local_deer_config.yaml