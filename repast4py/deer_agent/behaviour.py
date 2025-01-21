import random as rndm
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging"
from .movement import Movement, Point, Position_Vector
 
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

        "
        Both dispersal and exploratory movement are modeled with the same step lengths as normal movement 
        but with a turn angle distribution concentrated around zero
        "
    '''
    NORMAL = 1
    DISPERSE = 2
    MATING = 3
    EXPLORE = 4


@dataclass() 
class TimeOfYear_mixin:
    title: str
    start_day: int
    start_month: int
    end_day: int
    end_month: int

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


def calculate_next_state(agent):
    '''
    Based on current state, time, location etc. Calculate the next behavioural state
    of this agent.
    '''
    # if 



class TimeOfYear(TimeOfYear_mixin,Enum):
    '''
    - Gestation ( 1 Jan - 14 May)
    - Fawning   (15 May - 31 Aug)
    - PreRut    ( 1 Sep - 31 Oct)
    - Rut       ( 1 Nov - 31 Dec)
    '''
    GESTATION = 'Gestation', 1,1,14,5 
    FAWNING =   'Fawning',  15,5,31,8 
    PRERUT =    'PreRut',    1,9,31,10 
    RUT =       'Rut',       1,11,31,12 

# def check_time():
#     '''
#     Use step-time to calculate agent age and the the effects of this:
#         - Fawns growing up
#         - Old agents dying
#         - Pregnant females giving birth 

#     Check what time of year it is:
#     '''
#     self.age = tick_datetime - self.birth_date

#     if self.is_fawn and self.age > datetime.timedelta(days=params['deer_control_vars']['age']['fawn_to_adult']):
#         log.debug('Fawn grows up!')
#         self.is_fawn = False

#     if self.age > datetime.timedelta(days=params['deer_control_vars']['age']['adult_max']):
#         log.debug('Deer is too old!')
#         self.is_dead = True

#     time_of_year = 
#     return