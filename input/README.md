# Simulation Input

Files that are used to init the simulation, or to be used during the run are held here. 

## Rasters

This is the raster file to be loaded into the simulation. The rasters are projected into EPSG:5070 for a couple of reasons:

 - Pixel size becomes 30 x 30 meters and have equal areas across large regions.
 - Pixel are aligned East/West and North/South 
 - There isn't much distortion with latitude

The rasters are downloaded using the Notebook in the [../Notebooks](../Notebooks/README.md) folder.

## Building Files

## Agent Init Files