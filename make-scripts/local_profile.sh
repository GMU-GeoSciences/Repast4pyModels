#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "================================"
echo "== Profiling Repast Script... =="
echo "================================" 

echo "Running repast4py script in local docker env..."  
docker run -it --rm --name repast-local-docker -v "$PWD":/usr/src/myapp -w /usr/src/myapp repast-local-docker mpiexec -l -np 5 python -m cProfile -o ./profile.prof ./repast4py/deer_model.py ./config/local_deer_config.yaml
# https://stackoverflow.com/questions/33503176/profile-parallelized-python-script-with-mpi4py
# mpiexec -l -np 4 python -m cProfile ./simple-io.py doodad