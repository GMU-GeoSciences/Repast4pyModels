import random as rndm
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging" 

from . import behaviour

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
    
    return start_date <= date_stamp < end_date

def check_age(agent, params):
    '''
    Use step-time to calculate agent age and the the effects of this:
        - Fawns growing up
        - Old agents dying
        - Pregnant females giving birth 

    Check what time of year it is:
    '''
    agent.age = agent.timestamp - agent.birth_date
    fawn_to_adult = int(params['deer_control_vars']['age']['fawn_to_adult'])
    max_age = int(params['deer_control_vars']['age']['adult_max'])

    # Check if fawn grows up, and then maybe disperses.
    if agent.is_fawn and agent.age > datetime.timedelta(days=fawn_to_adult):
        log.debug(f'Fawn {agent.uuid} grows up!')
        agent.is_fawn = False
        male_disperse_prob = params['deer_control_vars']['age']['male_disperse_prob']
        female_disperse_prob = params['deer_control_vars']['age']['female_disperse_prob']
        if agent.is_male and (rndm.random() < male_disperse_prob):
            agent = behaviour.enter_disperse_state(agent)
        if not(agent.is_male) and (rndm.random() < female_disperse_prob):
            agent = behaviour.enter_disperse_state(agent) 

    # Check if deer is too old.
    if agent.age > datetime.timedelta(days=max_age):
        log.debug(f'Deer {agent.uuid} is too old!')
        agent.is_dead = True

    # Check if deer dies from some other event covered by "annual mortality"
    if agent.is_fawn:
        annual_mortality = params['deer_control_vars']['annual_mortality']['fawn']
    elif agent.is_male: 
        annual_mortality = params['deer_control_vars']['annual_mortality']['male']
    else: 
        annual_mortality = params['deer_control_vars']['annual_mortality']['female']
    tick_mortality = annual_mortality/365/24*int(params['time']['hours_per_tick'])
    
    if (rndm.random() < tick_mortality):
        log.debug(f'Deer {agent.uuid} has died from annual mortality rate!')
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
            return time_of_year
        else: 
            time_of_year = None
            log.warning(f'Time of year is unknown: {input_datetime}!')
            log.warning(f'Time of year is unknown: {input_datetime}!')
            return time_of_year

def increment_timers(agent):
    '''
    Increment all the timers once per step.
    '''
    agent.disease_timer += 1
    agent.gestation_timer += 1
    agent.behaviour_timer += 1
    agent.behaviour_max_timer += 1