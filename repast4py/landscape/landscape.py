import math 
from scipy.stats import exponweib, wrapcauchy
import numpy as np
from dataclasses import dataclass
import datetime 
import pandas as pd 
from repast4py.space import DiscretePoint as dpt
 
import logging as pylog # Repast logger is called as "logging"
log = pylog.getLogger(__name__)
'''
This file contains some functions for having agents interacting
with landscapes.
'''

def get_nearby_items(agent, model, sense_range = 100):
    '''
    Given a point, and an array, get the information for the local space.
    This is to be used to check whether the local area is good for 
    becoming a home range, or if the local area contains roads etc.

    Inputs:
        agent: the agent to examine
        model: the model that is being run. used to pull the landscape info 
        range: list: how near/far to check 
    Outputs: 
        local_array: the raster that surrounds the agent
        local_agents: list of other nearby agents within the sensor range
    '''

    # Find all the grid points that would fall into the "local area"
    # Loop through the grid points and do a repast call to get the shared_values
    # Calculate the stats of the shared_values. 
    # raster = np.ones([100,100])
    location = model.shared_space.get_location(agent)
    # grid_range = [math.ceil(meter_range / model.xy_resolution[0]) for meter_range in sense_range]
    grid_range = math.ceil(sense_range / model.xy_resolution[0])
     
    x_es = range(int(location.x) - grid_range, int(location.x) + grid_range)
    y_es = range(int(location.y) - grid_range, int(location.y) + grid_range)

    local_agents = []
    local_array = np.zeros(shape=(len(x_es), len(y_es)), dtype=int)
    for i in x_es:
        for j in y_es:
            x_ix = i - min(x_es) 
            y_ix = j - min(y_es)
            agents = model.grid.get_agents(dpt(i,j))
            for a in agents:
                if a == agent: # If the nearby agent is this agent do nothing 
                    pass
                else:
                    local_agents.append(a)
            local_array[x_ix,y_ix] = model.canopy_layer.get(dpt(i,j)).item()
            
    return local_array, local_agents



