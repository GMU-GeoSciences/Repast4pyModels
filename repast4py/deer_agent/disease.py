import random as rndm
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
import ast

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging" 

from .behaviour import *


log = pylog.getLogger(__name__)

'''
This hold the disease functions for the a S.I.R model.

Stuck in a state machine:
  - Susceptible
    - Move close to other infected
    - Become Infected
    - Randomly disease_end_time to some point in future
  - Infected 
    - Wait until current time > disease_end_time 
    - Become Recovered
    - Randomly disease_end_time to some point in future
  - Recovered 
    - Wait until current time > disease_end_time
    - Become Susceptible again
'''
 
class Disease_State(str, Enum):
  SUSCEPTIBLE = "Susceptible"
  INFECTED = "Infected"
  RECOVERED = "Recovered"

def become_infected(agent, params):
    '''
    What happens when the agent becomes infected.
    '''
    random_params = ast.literal_eval(params['deer_control_vars']['disease']['infection_duration'])
    infection_days = rndm.gauss(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=infection_days)
    agent.disease_state = Disease_State.INFECTED
    return agent

def become_recovered(agent, params):
    '''
    What happens when the agent becomes recovered.
    '''
    log.debug(f'    - Recovered! {agent.uuid}')

    random_params = ast.literal_eval(params['deer_control_vars']['disease']['immunity_duration'])
    immunity_days = rndm.gauss(random_params[0], random_params[1])

    agent.disease_end_datetime = agent.timestamp + datetime.timedelta(days=immunity_days)
    agent.disease_state = Disease_State.RECOVERED
    
    return agent

def become_susceptible(agent, params):
    '''
    What happens when the agent becomes susceptible.
    ''' 
    log.debug(f'    - Susceptible! {agent.uuid}')

    agent.disease_end_datetime = agent.timestamp
    agent.disease_state = Disease_State.SUSCEPTIBLE
    return agent


def check_infection_chance(agent, nearby_agents_list, params, resolution):
    '''
    Check if this agent has been infected by other agents.
    '''
    infection_chance = params['deer_control_vars']['disease']['infection_chance']
    infectious_range = float(params['deer_control_vars']['disease']['infectious_range'])
    # log.info(f'   -- {len(nearby_agents_list)} nearby other agents...')
    for other_agent in nearby_agents_list:
        distance_to_other = np.sqrt(np.square(agent.current_x - other_agent.current_x) + np.square(agent.current_y - other_agent.current_y))*resolution
        if (other_agent.disease_state == Disease_State.INFECTED): 
            if rndm.random() < infection_chance: 
                if distance_to_other < infectious_range:  
                    # log.info(f'    - Agent {other_agent.uuid} infected {agent.uuid} from {int(distance_to_other)} m!')
                    return True
    return False

def check_disease_state(agent, nearby_agents_list, params, resolution):
    '''
    Check the timers, check how the agents disease state progresses 
    '''
    log.debug(f'Checking disease status for agent {agent.uuid}. Current state = {str(agent.disease_state)}')
    if agent.disease_end_datetime < agent.timestamp:
        # Only do disease checks if enough time has passed since last state change: 
        if agent.disease_state == Disease_State.SUSCEPTIBLE:
            # Check nearby other agents
            if check_infection_chance(agent, nearby_agents_list, params, resolution):
                agent = become_infected(agent, params)
            return agent 
        
        if agent.disease_state == Disease_State.INFECTED:
            # Check timer to see if recovered
            agent = become_recovered(agent, params)
            return agent
        
        if agent.disease_state == Disease_State.RECOVERED:
            # Check timer to see if immunity has lapsed
            agent = become_susceptible(agent, params)
            return agent
    else:
        return agent
