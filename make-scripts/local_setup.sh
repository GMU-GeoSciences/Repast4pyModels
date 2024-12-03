#!/bin/bash
# This is to setup the Hopper environment in order to run the repast models
# It's a bit of a chicken/egg situation since you have to load git first
# in order to get this script in order to load git...

echo "=============================="
echo "== Setting-up Repast Env... =="
echo "==============================" 
echo "Setting up environment for local testing..."

echo "Pulling Container..."
docker pull ghcr.io/gmu-geosciences/repast4py-container:latest

echo "Installing anything new..."
docker build  -t repast-local-docker .

echo "Downloading raster files..."
# TODO: python script to cache rasters.

echo "============================="
echo "== Environment ready to go =="
echo "============================="