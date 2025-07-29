# Notebooks

This folder contains notebooks that have been used to investigate the data, or to generate inputs to the sim. They follow a general order 01,02, etc but can be run out of order to check different results, or to interate over an idea.

The notebooks are:

  - **01 GPS Preprocessing**: This notebook takes the raw GPS csv file, filters it, and then saves it into a parquet and geopackage files. The output of this notebook is used in the R notebook:
  - **02 HMM R Script**: This is a notebook that uses an R kernel instead of a Python kernel to train several HMM models using the filtered GPS data. The outputs of this are then **manually** inserted into a python movement model used in the simulation. 
  - **03 Fit Curves**: This notebook is not strictly required. It tries to fit curves to the GPS data histograms in order to get a feel for the data. This is more for a sanity check rather than to use it later in the simulation.
  - **Sim**: it's around about this point that you would run a simulation using a movement model that was taken from the R notebook.
  - **04 Post ABM Run**: This notebook is used to examine the repast agent log file that is create by an agent based model. It examines the step and turn movement histograms, the monthly home range created etc.
  - **05 Benchmark results**: This examines some factors that come out of a benchmark run: several simulations that have been run with varying configuration params. 