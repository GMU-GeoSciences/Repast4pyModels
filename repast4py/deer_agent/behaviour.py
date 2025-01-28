import random as rndm
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from copy import copy, deepcopy

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging" 
from . import time_functions
 
log = pylog.getLogger(__name__)

'''
This controls the behaviour state of the agents. Emulating Figure 2
from here: https://www.researchgate.net/publication/363077733 : 

    Furthermore,  the  following  deer behaviors  and attributes  heavily 
    infuence the above processes: 1) Individual movement at each time step, 
    as it  is dependent on  the age, sex,  and behavioral state  of the agent 
        - whether it is member of a group, tending to a fawn (female movement only),
        - following a  female  during  the  mating  season  (male  movement only), 
        - dispersing 
        - or conducting exploratory behavior 
    (See Appendix A and C for details on modeling movement based on empirical data), 2)
'''
 
class Behaviour_State(Enum):
    '''
    Limited set of behaviour states of this agent can have:
        - Normal: Do deer things
        - Disperse: Go in search of new home range
        - Mating: Follow a female during mating season
        - Explore: Go explore more. Less tight on home range
        "
        Exploratory movement lasts between 12-24 hours, the duration being chosen randomly as the 
        behavior starts, before the deer reverts to normal movements and returns to its home range.
        "
    '''
    NORMAL = 1
    DISPERSE = 2
    MATING = 3
    EXPLORE = 4
 
def local_suitability():
    '''
    Given local area and nearby agents, decide whether this is a good place to establish
    a home range, or whether this agent should rather disperse/explore
    '''
    
    return

def check_group():
    '''
    Check the group that this agent is part of: moms, siblings etc
    '''

def check_fawn():
    '''
    Get location of child and it's status. 
    '''

def check_disease():
    '''
    Check if sick, or recovered, and whether other nearby agents are sick.
    '''

def enter_explore_state(agent):
    '''
    when exploring the agent maintains it's centroid, changes it's movement angle params
    and has a limit on the explore duration:

    "
     Exploratory movement lasts between 12-24 hours, the duration being 
     chosen randomly as the behavior starts, before the deer reverts
     to normal movements and returns to its home range.
    "
    '''
    agent.behaviour_state = Behaviour_State.EXPLORE
    agent.explore_end_datetime = agent.timestamp + datetime.timedelta(hours=rndm.randint(12, 24))

    return agent

def enter_normal_state(agent):
    '''
    Normal state... Not much going on.
    '''
    agent.behaviour_state = Behaviour_State.NORMAL
    return agent


def enter_disperse_state(agent):
    '''
    Disperse! Ignore centroid and leave.
    '''
    log.debug(f'DISPERSE! {agent.uuid}')
    agent.behaviour_state = Behaviour_State.DISPERSE
    agent.has_homerange = False
    agent.explore_end_datetime = agent.timestamp + datetime.timedelta(hours=rndm.randint(12, 24))

    return agent

def enter_mating_state(agent):
    '''
    Mating!
    '''
    agent.behaviour_state = Behaviour_State.MATING 
    return agent

def establish_homerange(agent):
    '''
    This is a good place. Make it home
    '''
    log.debug(f'New home range for {agent.uuid}')
    agent.pos.centroid = copy(agent.pos.current_point)
    agent.has_homerange = True

    return agent

def grow_up(agent, params):
    '''
    Fawn grows up. This causes it to enter disperse (sometimes).
    '''
    agent.is_fawn = False

    # male_disperse_prob = params['deer_control_vars']['male_disperse_prob']
    # female_disperse_prob = params['deer_control_vars']['female_disperse_prob']
    # if agent.is_male and (rndm.random() < male_disperse_prob):
    #     agent = enter_disperse_state(agent)
    # if not(agent.is_male) and (rndm.random() < female_disperse_prob):
    #     agent = enter_disperse_state(agent)
    return agent

def location_suitability(local_array, nearby_agents, params):
    '''
    Check whether local conditions are good for the deer agent.
    '''
    pixel_threshold = float(params['deer_control_vars']['homerange']['suitability_threshold'])
    min_count = float(params['deer_control_vars']['homerange']['min'])
    good_pixels = (local_array > pixel_threshold).sum()

    return good_pixels > min_count


                    #########################
                    ## B E H A V I O U R S ##
                    ######################### 
def calculate_next_state(agent, local_cover, nearby_agents, params):
    '''
    Take current time of day, previous behaviour(s?), and location. 
    Use these to determine the behaviour of the deer. 

        Seeing as how this step is going to be run billions of times 
        computational efficiency is important. 
    '''  
    # time_of_year = time_functions.check_time_of_year(agent.timestamp)
    # agent = time_functions.check_age(agent, params) 
    good_hr = location_suitability(local_cover, nearby_agents, params) 

    ############### NORMAL ############### 
    if agent.behaviour_state == Behaviour_State.NORMAL:
        if agent.has_homerange:
            # sometimes start randomly exploring
            if not(good_hr):
                # agent.behaviour_state = Behaviour_State.DISPERSE
                pass
        else:
            # Agent has no homerange, go look for one.
            # agent.behaviour_state = Behaviour_State.DISPERSE 
            pass
        return agent
    
    ############### DISPERSE ###############
    elif agent.behaviour_state == Behaviour_State.DISPERSE:
        if agent.timestamp > agent.explore_end_datetime:
            # If Homerange is good stop exploring
            if good_hr:
                agent = enter_normal_state(agent)
                agent = establish_homerange(agent)
            else: 
                #Keep exploring a little longer
                agent = enter_disperse_state(agent)
        return agent
    
    # ############### MATING ###############
    # elif agent.behaviour_state == Behaviour_State.MATING:
    #     pass
    #     return agent
    
    # ############### EXPLORE ###############
    # elif agent.behaviour_state == Behaviour_State.EXPLORE:
    #     if agent.timestamp > agent.explore_end_datetime:
    #         if good_hr:
    #             # Stop exploring
    #             agent  = enter_normal_state(agent)
    #         else: 
    #             agent  = enter_normal_state(agent)

    #     else:
    #         # Keep Exploring
    #         pass 
    #     return agent

    log.warning('Something went wrong with behaviour step...')
    return agent
#######################################################################
#######################################################################