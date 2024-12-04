#!/bin/bash
# This cleans up the local environment
echo "=============================="
echo "== Cleaning up files...     =="
echo "==============================" 

echo "Deleting log files..." 
rm -i ./output/*.csv

echo "Deleting singularity image..." 
rm -i /containers/hopper/UserContainers/$USER/repast4py_latest.sif
# docker rmi $(docker images repast-test)

# echo "Deleting raster images..."
# rm ./input/images/*.tiff
