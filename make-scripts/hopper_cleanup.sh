#!/bin/bash
# This cleans up the local environment
echo "=============================="
echo "== Cleaning up files...     =="
echo "==============================" 

echo "Deleting log files..." 
rm -i ./output/*.csv

echo "Deleting singularity image..." 
rm -i /containers/hopper/UserContainers/$USER/repast4py_latest.sif

echo "Deleting raster images..."
rm -i ./input/images/*.tiff

echo "Unloading modules..."
module unload singularity openmpi4

echo "============================="
echo "==  Environment is clean   =="
echo "============================="