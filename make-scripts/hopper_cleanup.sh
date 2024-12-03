#!/bin/bash
# This cleans up the local environment
echo "=============================="
echo "== Cleaning up files...     =="
echo "==============================" 

echo "Deleting log files..." 
rm ./output/*.csv

echo "TODO: Deleting singularity image..." 
# docker rmi $(docker images repast-test)

echo "Deleting raster images..."
# rm ./input/images/*.tiff
