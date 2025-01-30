#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "=============================="
echo "== Running Repast Script... =="
echo "==============================" 

echo "Running repast4py script in local docker env..."
EXP="spatial_area"

echo "====================="
echo "== Benchmark 01... =="
echo "=====================" 
docker run -it --rm --name repast-local-docker -v "$PWD":/usr/src/myapp -w /usr/src/myapp repast-local-docker mpirun -n 6 python ./repast4py/deer_model.py ./config/experiments/$EXP/01_benchmark_config.yaml

echo "====================="
echo "== Benchmark 02... =="
echo "=====================" 
docker run -it --rm --name repast-local-docker -v "$PWD":/usr/src/myapp -w /usr/src/myapp repast-local-docker mpirun -n 6 python ./repast4py/deer_model.py ./config/experiments/$EXP/02_benchmark_config.yaml

echo "====================="
echo "== Benchmark 03... =="
echo "=====================" 
docker run -it --rm --name repast-local-docker -v "$PWD":/usr/src/myapp -w /usr/src/myapp repast-local-docker mpirun -n 6 python ./repast4py/deer_model.py ./config/experiments/$EXP/03_benchmark_config.yaml

echo "====================="
echo "== Benchmark 04... =="
echo "=====================" 
docker run -it --rm --name repast-local-docker -v "$PWD":/usr/src/myapp -w /usr/src/myapp repast-local-docker mpirun -n 6 python ./repast4py/deer_model.py ./config/experiments/$EXP/04_benchmark_config.yaml

echo "==========="
echo "== Done! =="  
echo "===========" 