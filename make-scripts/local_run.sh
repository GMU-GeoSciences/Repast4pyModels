#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "=============================="
echo "== Running Repast Script... =="
echo "==============================" 

echo "Running repast4py script in local docker env..."  
docker run -it --rm --name repast-local-docker -e PYTHONFAULTHANDLER=1 -v "$PWD":/usr/src/myapp -w /usr/src/myapp repast-local-docker mpirun -n 3 python ./repast4py/deer_model.py ./config/local_deer_config.yaml

# PYTHONFAULTHANDLER=1 python myscript.py