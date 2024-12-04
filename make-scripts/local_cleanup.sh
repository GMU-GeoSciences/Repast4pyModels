#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "=============================="
echo "== Cleaning up files...     =="
echo "==============================" 

echo "Deleting log files..." 
rm ./output/*.csv

echo "Deleting docker image..." 
docker rmi $(docker images -q repast-local-docker)

# echo "Deleting raster images..."
# rm ./input/images/*.tiff
