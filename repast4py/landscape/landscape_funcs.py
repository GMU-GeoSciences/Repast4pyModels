import math 
from scipy.stats import exponweib, wrapcauchy
import numpy as np
from dataclasses import dataclass
import datetime 
import pandas as pd 
from repast4py.space import DiscretePoint as dpt
from repast4py.geometry import BoundingBox
import functools
 
import logging as pylog # Repast logger is called as "logging"
log = pylog.getLogger(__name__)
'''
This file contains some functions for having agents interacting
with landscapes.
'''

@functools.cache
def get_pixel_value(canopy_layer, i,j):
    '''
    Given a repast discrete point, find the associated raster shared value.
    Since the raster value never changed this function can be cached.
    ''' 
    return canopy_layer.get(dpt(i,j)).item()

def get_nearby_grid_agents(agent, model, sense_range = 100):
    '''
    Get all agents at the discrete point(i,j) that
    are not the same as "this_agent"
    ''' 
    # agents = model.grid.get_agents(dpt(i,j))
    # agents = [agent for agent in agents if not this_agent] 
    log.debug('  -Getting nearby other agents...')
    grid_range = math.ceil(sense_range / model.xy_resolution[0]) #Turn meters into pixels

    location = model.shared_space.get_location(agent)
    bbox = BoundingBox(int(location.x) - grid_range, 2*grid_range,
                       int(location.y) - grid_range, 2*grid_range)
    log.debug(f'  -Local BBOX: {bbox}')
    other_agents = model.shared_space.get_agents_within(bbox)
    other_agents = [other_agent for other_agent in other_agents if other_agent != agent] 
 
    return other_agents

def get_nearby_pixels(agent, model, sense_range = 100):
    '''
    Given a point, and an array, get the information for the local space.
    This is to be used to check whether the local area is good for 
    becoming a home range, or if the local area contains roads etc.

    Inputs:
        agent: the agent to examine
        model: the model that is being run. used to pull the landscape info 
        sense_range: how many meters to check around the agent
    Outputs: 
        local_array: the raster that surrounds the agent
        local_agents: list of other nearby agents within the sensor range

    From cProfile: About 25% of the time is spent in this function.
    '''
    grid_range = math.ceil(sense_range / model.xy_resolution[0]) #Turn meters into pixels

    location = model.shared_space.get_location(agent)
    # bbox = BoundingBox(int(location.x) - grid_range, 2*grid_range,
    #                    int(location.y) - grid_range, 2*grid_range)
    
    # other_agents = model.shared_space.get_agents_within(bbox)
    # local_agents = [other_agent for other_agent in other_agents if not agent] 
     
    x_es = range(int(location.x) - grid_range, int(location.x) + grid_range)
    y_es = range(int(location.y) - grid_range, int(location.y) + grid_range)

    # local_agents = []
    local_array = np.zeros(shape=(len(x_es), len(y_es)), dtype=int) # Empty array for sense_range pixels around the agent
    
    # agents = model.shared_space.get_agents_within(bbox)
    # TODO: This could probably be sped up using map() or anything other than
    # a nested for loop...
    #https://medium.com/@nirmalya.ghosh/13-ways-to-speedup-python-loops-e3ee56cd6b73 
    for i in x_es:
        for j in y_es: 
            # local_agents.extend(get_nearby_agents(model, agent, i,j)) 
            local_array[i - min(x_es), j - min(y_es)] = get_pixel_value(model.canopy_layer, i,j)

    return local_array 
 
def get_nearby_agents(model, this_agent, i,j):
    '''
    Get all agents at the discrete point(i,j) that
    are not the same as "this_agent"
    '''
    agents_in_cell = []
    agents = model.grid.get_agents(dpt(i,j))
    for a in agents:
        if a == this_agent: # If the nearby agent is this agent do nothing 
            pass
        else:
            agents_in_cell.append(a)
    return agents_in_cell

def get_nearby_items(agent, model, sense_range = 100):
    '''
    Given a point, and an array, get the information for the local space.
    This is to be used to check whether the local area is good for 
    becoming a home range, or if the local area contains roads etc.

    Inputs:
        agent: the agent to examine
        model: the model that is being run. used to pull the landscape info 
        sense_range: how many meters to check around the agent
    Outputs: 
        local_array: the raster that surrounds the agent
        local_agents: list of other nearby agents within the sensor range

    From cProfile: About 60% of the time is spent in this function.
    '''
    location = model.shared_space.get_location(agent)
    grid_range = math.ceil(sense_range / model.xy_resolution[0]) #Turn meters into pixels
     
    x_es = range(int(location.x) - grid_range, int(location.x) + grid_range)
    y_es = range(int(location.y) - grid_range, int(location.y) + grid_range)

    local_agents = []
    local_array = np.zeros(shape=(len(x_es), len(y_es)), dtype=int) # Empty array for sense_range pixels around the agent
    
    # TODO: This could probably be sped up using map() or anything other than
    # a nested for loop...
    #https://medium.com/@nirmalya.ghosh/13-ways-to-speedup-python-loops-e3ee56cd6b73
    for i in x_es:
        for j in y_es: 
            local_agents.extend(get_nearby_agents(model, agent, i,j)) 
            local_array[i - min(x_es), j - min(y_es)] = get_pixel_value(model.canopy_layer, i,j)

    return local_array, local_agents
 