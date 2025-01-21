import random as rndm
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging" 
 
log = pylog.getLogger(__name__)

'''
This hold the time functions. Nothing too fancy but just keeping
it separate for readability 
'''
  
@dataclass() 
class TimeOfYear_mixin:
    title: str
    start_day: int
    start_month: int
    end_day: int
    end_month: int
 
class DeerSeasons(TimeOfYear_mixin, Enum):
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

def is_season(timestamp, deer_season):
    '''
    Returns true if timestamp falls between start and end date of deer_season
    '''
    date_stamp = timestamp.date()
    start_date = datetime.date(year = date_stamp.year,
                                    month = deer_season.start_month,
                                    day = deer_season.start_day)
    
    end_date = datetime.date(year = date_stamp.year,
                                    month = deer_season.end_month,
                                    day = deer_season.end_day)
    
    return start_date < timestamp < end_date

def check_age(input_datetime, agent, params):
    '''
    Use step-time to calculate agent age and the the effects of this:
        - Fawns growing up
        - Old agents dying
        - Pregnant females giving birth 

    Check what time of year it is:
    '''
    agent.age = input_datetime - agent.birth_date
    fawn_to_adult = int(params['deer_control_vars']['age']['fawn_to_adult'])
    max_age = int(params['deer_control_vars']['age']['adult_max'])

    if agent.is_fawn and agent.age > fawn_to_adult:
        log.debug('Fawn grows up!')
        agent.is_fawn = False

    if agent.age > datetime.timedelta(days=max_age):
        log.debug('Deer is too old!')
        agent.is_dead = True

    return agent

def check_time_of_year(input_datetime):
    '''
    Check what time period it is in this tick.
    '''
    for season in DeerSeasons:
        # loop through DeerSeasons enum and return the 
        # season that envelopes input_datetime.
        if is_season(input_datetime, season):
            time_of_year = season
        else: 
            time_of_year = None

    return time_of_year