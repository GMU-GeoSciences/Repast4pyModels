# Repast4pyModels

## Code Structure

The code is structured a little messily, if you're the poor soul that is going to use/refactor this please get hold of me with any questions that you have. I'll try my best to remember what I was thinking when I wrote the code and try to help out.

But here's a quick breakdown:

  - *deer_model.py*: This file defines the repast model, how data get's logged, and parses the config file in order to configure and setup the simulation
  - *landscape folder*: This contains a submodule that deals with the GIS portion of the data, it fetches (or reads) a geotiff file for the region of interest, it gets nearby items in the simulation. NOTE: there is some conversion between repast4py arrays and 5070 projected coords in the agent object. Organic code growth, what can I say?
  - *deer_agent folder*: This contains code for the agents, movement models, SIR disease models, and time functions. The real meat-and-potatoes of the simulation.

## How does the code flow actually work

So after the docker/singularity environment is setup and the config applied the python code goes through:

  - Parsing the config.yaml into a python dictionary
  - Setting up logging library
  - The repast Model object is initialised:
    - Start/stop is scheduled
    - Setup the spatial array and coord system for the simulation
    - Setup the agent logging (the actual out put as opposed to the std-out logging. )
    - the agents are initialised
  - The sim is started.
  - The sim is stepped through each time step, agents are moved around, steps are logged.
  - The sim is ended.


## Repast Spatial

The method used to contain spatial information, specifically a raster. 