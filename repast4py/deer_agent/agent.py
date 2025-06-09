import random as rndm

from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

import datetime
import numpy as np
import uuid
import ast

import logging as pylog # Repast logger is called as "logging"
from .movement import *
from .behaviour import *
from .disease import *

log = pylog.getLogger(__name__)

def get_new_uuid() -> str:
    return str(uuid.uuid4())
 
@dataclass
class DiseaseState:
  name: str
  template: "DiseaseEnum"

class DiseaseEnum(str, Enum):
  SUSCEPTIBLE = "Susceptible"
  INFECTED = "Infected"
  RECOVERED = "Recovered"
 
@dataclass
class Deer_Config(object):
    '''
    Some basic configuration for a deer agent. 
    '''  
    uuid: str = field(default_factory=get_new_uuid) 
    random_seed: int = 0
    group_id: int = 0
    birth_date: datetime = datetime.datetime(2020, 1, 1)
    is_male: bool = False
    is_fawn: bool = False
    gestation_timer: int = 0
    has_homerange: bool = False # Assume it's false and check that initial pos is valid for centroid
    behaviour_state: behaviour.Behaviour_State = field(default = behaviour.Behaviour_State.DISPERSE) 
    disease_state: Disease_State = field(default = Disease_State.SUSCEPTIBLE) 
    behaviour_timer: int = 0
    explore_end_datetime: datetime = datetime.datetime(2020, 1, 2)
    disease_end_datetime: datetime = datetime.datetime(2020, 1, 2)
    is_dead: bool = False
    # pos: Position_Vector = field(default_factory = lambda:Position_Vector(Point(0, 0),
    #                                                                          Point(0, 0),
    #                                                                          Point(0, 0)))
    current_x: float = 0.0 # field(repr=False, default=10)
    current_y: float = 0.0
    prev_x: float = 0.0
    prev_y: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    heading_from_prev: float = 0.0
    heading_to_centroid: float = 0.0
    step_dist: float = 0.0
    turn_angle: float = 0.0
    timestamp: datetime.datetime = datetime.datetime.fromisoformat('1970-01-01')

    def __post_init__(self):
        '''
        Assume the start point of all deer agents
        is also their last/current point and the centroid
        '''
        last_point = Point(self.current_x, self.current_y)
        current_point = Point(self.current_x, self.current_y)
        centroid = Point(self.current_x, self.current_y)
        self.pos = Position_Vector(last_point, current_point, centroid, 0 ,0) 
    
    @classmethod
    def rand_factory(cls, x, y, params):
        '''
        Returns a randomised deer agent config
        '''
        initial_x = x
        initial_y = y
        initial_timestamp = datetime.datetime.fromisoformat(params['time']['start_time'])
        initial_infection_chance = float(params['deer_control_vars']['disease']['infectious_start_rate'])
        
        if rndm.random() < initial_infection_chance:
            # Agent should be infected 
            random_params = ast.literal_eval(params['deer_control_vars']['disease']['immunity_duration'])
            immunity_days = rndm.gauss(random_params[0], random_params[1]) 
            disease_end_datetime = initial_timestamp + datetime.timedelta(days=immunity_days)
            disease_state = Disease_State.INFECTED
        else: 
            disease_state = Disease_State.SUSCEPTIBLE
            disease_end_datetime = initial_timestamp

        return cls(
            uuid  = get_new_uuid(), 
            random_seed = rndm.randint(1, 100000),
            group_id = None,
            birth_date = initial_timestamp - datetime.timedelta(days=rndm.randint(1, 2000)),
            is_male = rndm.random() < 0.5,
            is_fawn = rndm.random() < 0.2,
            gestation_timer = 0,
            has_homerange = False,  
            behaviour_state = behaviour.Behaviour_State.DISPERSE,
            disease_state = disease_state,
            behaviour_timer = 0,
            explore_end_datetime = initial_timestamp + datetime.timedelta(hours=rndm.randint(12, 24)),
            disease_end_datetime = disease_end_datetime,
            is_dead = False,
            # pos = Position_Vector(last_point = Point(initial_x, initial_y),
            #                       current_point = Point(initial_x, initial_y),
            #                       centroid = Point(initial_x, initial_y) ),
            current_x = initial_x,
            current_y = initial_y,
            prev_x = 0,
            prev_y = 0,
            centroid_x = 0.0,
            centroid_y =0.0,
            heading_from_prev = 0.0,
            heading_to_centroid = 0.0,
            step_dist = 0,
            turn_angle = 0,
            timestamp = initial_timestamp,)

def classFromArgs(className, argDict):
    '''
    https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary
    '''
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)

class DeerAgent(core.Agent):
    """The Deer Agent.

    Currently the agent has a starting centroid and does random a walk, always staying near to centroid. 

    Args:
        a_id: a integer that uniquely identifies this agent on its starting rank
        rank: the starting MPI rank of this agent.
    """
    # TYPE is a class variable that defines the agent type id the Neuron agent. This is a required part of the unique agent id tuple.
    TYPE = 0

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=DeerAgent.TYPE, rank=rank)

    def set_cfg(self, agent_cfg):
        '''
        Set/Get the config variables to/from a dict. This is to avoid the agent restore 
        function having a bunch of tuples.
        '''
        self.uuid = agent_cfg.uuid 
        self.random_seed = agent_cfg.random_seed
        self.group_id = agent_cfg.group_id
        self.birth_date = agent_cfg.birth_date
        self.is_male = agent_cfg.is_male
        self.is_fawn = agent_cfg.is_fawn
        self.gestation_timer = agent_cfg.gestation_timer
        self.has_homerange = agent_cfg.has_homerange
        self.behaviour_state = agent_cfg.behaviour_state
        self.disease_state = agent_cfg.disease_state 
        self.behaviour_timer = agent_cfg.behaviour_timer
        self.explore_end_datetime = agent_cfg.explore_end_datetime
        self.disease_end_datetime = agent_cfg.disease_end_datetime
        self.is_dead = agent_cfg.is_dead
        self.current_x = agent_cfg.current_x
        self.current_y = agent_cfg.current_y
        self.prev_x = agent_cfg.prev_x
        self.prev_y = agent_cfg.prev_y
        self.centroid_x = agent_cfg.centroid_x
        self.centroid_y = agent_cfg.centroid_y
        self.heading_from_prev = agent_cfg.heading_from_prev
        self.heading_to_centroid = agent_cfg.heading_to_centroid
        self.step_dist = agent_cfg.step_dist
        self.turn_angle = agent_cfg.turn_angle
        self.timestamp = agent_cfg.timestamp

    def get_cfg(self):
        agent_cfg = Deer_Config()
        agent_cfg.uuid = self.uuid  
        agent_cfg.random_seed = self.random_seed 
        agent_cfg.group_id = self.group_id  
        agent_cfg.birth_date = self.birth_date  
        agent_cfg.is_male = self.is_male
        agent_cfg.is_fawn = self.is_fawn  
        agent_cfg.gestation_timer = self.gestation_timer  
        agent_cfg.has_homerange = self.has_homerange  
        agent_cfg.current_x = self.current_x 
        agent_cfg.current_y = self.current_y 
        agent_cfg.behaviour_state = self.behaviour_state 
        agent_cfg.disease_state = self.disease_state
        agent_cfg.behaviour_timer = self.behaviour_timer
        agent_cfg.explore_end_datetime = self.explore_end_datetime
        agent_cfg.disease_end_datetime = self.disease_end_datetime
        agent_cfg.is_dead = self.is_dead
        agent_cfg.current_x = self.current_x
        agent_cfg.current_y = self.current_y
        agent_cfg.prev_x = self.prev_x
        agent_cfg.prev_y = self.prev_y
        agent_cfg.centroid_x = self.centroid_x
        agent_cfg.centroid_y = self.centroid_y
        agent_cfg.heading_from_prev = self.heading_from_prev
        agent_cfg.heading_to_centroid = self.heading_to_centroid  
        agent_cfg.step_dist = self.step_dist
        agent_cfg.turn_angle = self.turn_angle
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
        log.debug('Saving agent to move to different rank...')
        cfg = self.get_cfg()
        log.debug(cfg)
        return (self.uid,
                cfg)
     
    def calc_next_point(self, step_distance, turn_angle):
        '''
        When given a distance and angle calculate the X and Y coords of it
        when starting from a current position. 

        Turn Angle = Angle(Prev, Current) - Angle(Current, Next)
        ''' 
        next_x = self.current_x + step_distance*np.sin(self.heading_from_prev + turn_angle)
        next_y = self.current_y + step_distance*np.cos(self.heading_from_prev + turn_angle)
        
        # Update all the class items:
        self.prev_x = self.current_x
        self.prev_y = self.current_y

        self.current_x = next_x
        self.current_y = next_y

        a = np.array((self.current_x, self.current_y))
        b = np.array((self.centroid_x, self.centroid_y))
        self.distance_to_centroid = np.linalg.norm(a-b) # Euclidian distance from current location to centroid. NOTE: this might be in repast grid units not meters.

        dx = self.centroid_x - self.current_x
        dy = self.centroid_y - self.current_y
        angle_radians = np.arctan2(dx, dy)
        self.heading_to_centroid = (angle_radians) % (2 * np.pi) # Compass direction of travel between current_pos and centroid 

        dx = self.current_x - self.prev_x
        dy = self.current_y - self.prev_y
        angle_radians = np.arctan2(dx, dy)
        self.heading_from_prev = (angle_radians) % (2 * np.pi) # Compass direction of travel between last_pos and current_pos

        self.step_dist = step_distance
        self.turn_angle = turn_angle
        return 

    # def calculate_next_pos(self, xy_resolution):
    #     '''
    #     Take last known position, home range centroid, and average_speed
    #     and then calculate the location for the next step. Calculations 
    #     change based on:
    #      - behaviour state: foraging/resting travels less than others
    #      - distance away from centroid: animals prefer to stay close to home range
    #      - average speed: faster animals move faster... 

    #      Seeing as how this step is going to be run billions of times 
    #      computational efficiency is important. 
    #     ''' 
    #     # CurrentX/Y should be the same as self.pos.current_point
    #     # log.debug(f'Pos Vector: {self.pos}') 
    #     next_position = movement.step(self, xy_resolution) 
    #     return next_position.x, next_position.y