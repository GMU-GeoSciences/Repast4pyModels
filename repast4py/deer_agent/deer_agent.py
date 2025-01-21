import random as rndm

from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

import datetime
import numpy as np

import logging as pylog # Repast logger is called as "logging"
from .movement import Movement, Point, Position_Vector
from .behaviour import Behaviour_State

log = pylog.getLogger(__name__)
 
@dataclass
class Deer_Config(object):
    '''
    Some basic configuration for a deer agent. 
    ''' 
    is_infected: bool = False
    is_contagious: bool = False
    has_recovered: bool = False
    disease_timer: int = 0
    random_seed: int = 0
    group_id: int = 0
    birth_date: datetime = datetime.datetime(2020, 1, 1)
    is_male: bool = False
    is_fawn: bool = False
    gestation_days: int = 0
    has_homerange: bool = False # Assume it's false and check that initial pos is valid for centroid
    current_x: float = 0.0 # field(repr=False, default=10)
    current_y: float = 0.0
    behaviour_state: Behaviour_State = field(default = Behaviour_State.NORMAL) 
    is_dead: bool = False
    pos: Position_Vector = field(default = Behaviour_State.NORMAL)
    timestamp: datetime.datetime = datetime.datetime.fromisoformat('1970-01-01')

    def __post_init__(self):
        '''
        Assume the start point of all deer agents
        is also their last/current point and the centroid
        '''
        last_point = Point(self.current_x,self.current_y)
        current_point = Point(self.current_x,self.current_y)
        centroid = Point(self.current_x, self.current_y)
        self.pos = Position_Vector(last_point, current_point, centroid) 
    
    @classmethod
    def rand_factory(cls, x, y, params):
        '''
        Returns a randomised deer agent config
        '''
        initial_x = x
        initial_y = y
        initial_timestamp = datetime.datetime.fromisoformat(params['time']['start_time']) 

        return cls(
            is_infected   = rndm.random() < params['deer']['disease_starting_ratio'], # Is true if random number below 0.1. Thus ~10% of population will start infected...  
            is_contagious = False, 
            has_recovered = False,
            disease_timer = rndm.randint(1, 100),
            random_seed = rndm.randint(1, 100000),
            group_id = None,
            birth_date = initial_timestamp - datetime.timedelta(days=rndm.randint(1, 2000)),
            is_male = rndm.random() < 0.5,
            is_fawn = rndm.random() < 0.2,
            gestation_days = 0,
            has_homerange = False, 
            current_x = initial_x,
            current_y = initial_y,
            behaviour_state = Behaviour_State.NORMAL, 
            is_dead = False,
            pos = Position_Vector(last_point = Point(initial_x, initial_y),
                                  current_point = Point(initial_x, initial_y),
                                  centroid = Point(initial_x, initial_y)),
            timestamp = initial_timestamp,)

def classFromArgs(className, argDict):
    '''
    https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary
    '''
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)

class Deer(core.Agent):
    """The Deer Agent.

    Currently the agent has a starting centroid and does random a walk, always staying near to centroid. 

    Args:
        a_id: a integer that uniquely identifies this agent on its starting rank
        rank: the starting MPI rank of this agent.
    """
    # TYPE is a class variable that defines the agent type id the Neuron agent. This is a required part of the unique agent id tuple.
    TYPE = 0

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Deer.TYPE, rank=rank)

    def set_cfg(self, agent_cfg):
        '''
        Set/Get the config variables to/from a dict. This is to avoid the agent restore 
        function having a bunch of tuples.
        '''
 
        # Infection Vars:
        self.is_infected = agent_cfg.is_infected
        self.is_contagious = agent_cfg.is_contagious
        self.has_recovered = agent_cfg.has_recovered
        self.disease_timer = agent_cfg.disease_timer
        self.random_seed = agent_cfg.random_seed
        self.group_id = agent_cfg.group_id
        self.birth_date = agent_cfg.birth_date
        self.is_male = agent_cfg.is_male
        self.is_fawn = agent_cfg.is_fawn
        self.gestation_days = agent_cfg.gestation_days
        self.has_homerange = agent_cfg.has_homerange
        self.current_x = agent_cfg.current_x
        self.current_y = agent_cfg.current_y
        self.behaviour_state = agent_cfg.behaviour_state
        self.is_dead = agent_cfg.is_dead

        self.pos = agent_cfg.pos
        self.timestamp = agent_cfg.timestamp  

    def get_cfg(self):
        agent_cfg = Deer_Config()

        agent_cfg.is_infected = self.is_infected 
        agent_cfg.is_contagious = self.is_contagious 
        agent_cfg.has_recovered = self.has_recovered  
        agent_cfg.disease_timer = self.disease_timer  
        agent_cfg.random_seed = self.random_seed 
        agent_cfg.group_id = self.group_id  
        agent_cfg.birth_date = self.birth_date  
        agent_cfg.is_male = self.is_male
        agent_cfg.is_fawn = self.is_fawn  
        agent_cfg.gestation_days = self.gestation_days  
        agent_cfg.has_homerange = self.has_homerange  
        agent_cfg.current_x = self.current_x 
        agent_cfg.current_y = self.current_y 
        agent_cfg.behaviour_state = self.behaviour_state 
        agent_cfg.is_dead = self.is_dead

        agent_cfg.pos = self.pos
        agent_cfg.timestamp = self.timestamp  
        
        return agent_cfg

    def save(self) -> Tuple:
        """Saves the state of this agent as a Tuple.

        Used to move this agent from one MPI rank to another.

        Returns:
            The saved state of this agent:
            (uid tuple,
             movement object,
             config dict
             )

        Example: ((5, 0, 1), False, 0, (0, 0), 2, 1, 0)
        """ 
        cfg = self.get_cfg()
        log.debug(cfg)
        return (self.uid,
                cfg)

    def expose_to_infection(self, tick):
        '''
        Deer is exposed to infection. State changes depend on immunity, if already sick, etc 
        '''
        self.is_infected = True
        self.disease_timer = 0 
        # TODO: This needs to be more complex
        return

    def calculate_next_pos(self, current_x, current_y, tick_datetime):
        '''
        Take last known position, home range centroid, and average_speed
        and then calculate the location for the next step. Calculations 
        change based on:
         - behaviour state: foraging/resting travels less than others
         - distance away from centroid: animals prefer to stay close to home range
         - average speed: faster animals move faster... 

         Seeing as how this step is going to be run billions of times 
         computational efficiency is important. 
        ''' 
        # CurrentX/Y should be the same as self.pos.current_point
        log.debug(f'Pos Vector: {self.pos}')
        agent_movement = Movement(self.pos, tick_datetime)
        next_position = agent_movement.step() 
        return next_position.x, next_position.y
    
    def calculate_next_state(self):
        '''
        Take current time of day, previous behaviour(s?), and location. 
        Use these to determine the behaviour of the deer. 

         Seeing as how this step is going to be run billions of times 
         computational efficiency is important. 
        '''
        # TODO: This needs to be more complex
        self.behaviour_state = 1
        self.disease_timer += 1 
        return
    
    def check_age(self, tick_datetime, params):
        '''
        Update age from birthdate + tick
        Check what time of year it is:
            - Gestation ( 1 Jan - 14 May)
            - Fawning   (15 May - 31 Aug)
            - PreRut    ( 1 Sep - 31 Oct)
            - Rut       ( 1 Nov - 31 Dec)
        '''
        self.age = tick_datetime - self.birth_date
        fawn_to_adult = int(params['deer_control_vars']['age']['fawn_to_adult'])
        max_age = int(params['deer_control_vars']['age']['adult_max'])

        if self.is_fawn and self.age > datetime.timedelta(days=fawn_to_adult):
            log.debug('Fawn grows up!')
            self.is_fawn = False

        if self.age > datetime.timedelta(days=max_age):
            log.debug('Deer is too old!')
            self.is_dead = True

        return
    
    def check_homerange(self, local_cover, params):
        '''
        If in dispersal mode check if this can be a home range
        If in normal mode recalc direction to centroid
        '''
        
        # Dummy test. Make current position the centroid no matter what
        ###################################################
        self.has_home_range = False # Agent has a home range.
        ###################################################
        
        if self.has_home_range:
            pass
            # TODO:
            # Has a home range. Should probably check if other deer are around?
        else: 
            if (local_cover > params['deer_control_vars']['homerange']['suitability_threshold']).sum() > params['deer_control_vars']['homerange']['min']:
                # Local area has potential to be home range.
                self.has_home_range = True
                self.pos.centroid = self.pos.current_point
                self.behaviour_state = Behaviour_State.NORMAL
            else:
                # Location is not good for home range
                self.behaviour_state = Behaviour_State.EXPLORE
        return
    
    def check_group(self):
        '''
        Check distance to herd group, distance to fawn, or distance to potential mate
        '''

        return

    def check_disease(self, tick_datetime):
        '''
        Check disease progression, whether this agent is infectious 
        '''
        return

    def calculate_next_state(self):
        '''
        Check whether the next state of this agent should be:
            - Normal: Do deer things
            - Disperse: Go in search of new home range
            - Mating: Follow a female during mating season
            - Explore: Go explore more. Less tight on home range
        '''
        return