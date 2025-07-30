import random as rndm

from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
from typing import Tuple 
from dataclasses import dataclass, field, fields, asdict
from enum import Enum

import datetime
import numpy as np
import math
import uuid
import ast

import logging as pylog # Repast logger is called as "logging"
# from .movement import *
from .behaviour import *
from .disease import *
from . import behaviour, time_functions

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
    distance_to_centroid: float =0.0
    step_dist: float = 0.0
    turn_angle: float = 0.0
    timestamp: datetime.datetime = datetime.datetime.fromisoformat('1970-01-01')

    # def __post_init__(self):
    #     '''
    #     Assume the start point of all deer agents
    #     is also their last/current point and the centroid
    #     '''
        # last_point = Point(self.current_x, self.current_y)
        # current_point = Point(self.current_x, self.current_y)
        # centroid = Point(self.current_x, self.current_y)
        # self.pos = Position_Vector(last_point, current_point, centroid, 0 ,0) 
    
    @classmethod
    def rand_factory(cls, x, y, params):
        '''
        Returns a randomised deer agent config
        '''
        initial_x = x
        initial_y = y
        initial_timestamp = datetime.datetime.fromisoformat(params['time']['start_time'])
        initial_infection_chance = float(params['deer_control_vars']['disease']['infectious_start_rate'])
        male_proportion = params['deer']['male_proportion']
        
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
            is_male = rndm.random() < male_proportion,  
            is_fawn = rndm.random() < 0.2,
            gestation_timer = 0,
            has_homerange = False,  
            behaviour_state = behaviour.Behaviour_State.DISPERSE,
            disease_state = disease_state,
            behaviour_timer = 0,
            explore_end_datetime = initial_timestamp + datetime.timedelta(hours=rndm.randint(12, 24)),
            disease_end_datetime = disease_end_datetime,
            is_dead = False, 
            current_x = initial_x,
            current_y = initial_y,
            prev_x = 0,
            prev_y = 0,
            centroid_x = 0.0,
            centroid_y =0.0,
            heading_from_prev = 0.0,
            heading_to_centroid = 0.0,
            distance_to_centroid = 0.0,
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
        self.distance_to_centroid = agent_cfg.distance_to_centroid
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
        agent_cfg.distance_to_centroid = self.distance_to_centroid
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
    
    def proj_to_repast(self, x_proj, y_proj,
                        xy_resolution, image_bounds):
        '''
        Take repast grid x,y position and 
        turn it into a 5070 projection

        bbox= {'left':1595925.0, 
                'bottom':1641105.0, 
                'right':1952145.0, 
                'top':1979385.0})
        '''   
        log.debug(f'--- Change Repast Grid to 5070 Projection using xy_resolution: {xy_resolution}')
        x_repast = (x_proj - image_bounds.left)/xy_resolution[0]
        y_repast = -(y_proj- image_bounds.top )/xy_resolution[1]
    
        return x_repast, y_repast
    
    def repast_to_proj(self, x_repast, y_repast, 
                        xy_resolution, image_bounds):
        '''
        Take repast grid x,y position and 
        turn it into a 5070 projection
        ''' 
        log.debug('--- Change 5070 Projection to Repast Grid')
        x_proj = x_repast*xy_resolution[0] + image_bounds.left 
        y_proj = image_bounds.top - y_repast*xy_resolution[1]
        return x_proj, y_proj
    
    def bearing_to_turn_angle(self, bearing):
        '''
        Take a bearing and convert it into the turn 
        required to reach that bearing
        '''
        log.debug('--- Calc Bearing from Angles')
        turn_angle = bearing
        return turn_angle
    
    def turn_angle_to_bearing(self, turn_angle):
        '''
        Take a turn angle and convert it into
        compass bearing.
        '''
        log.debug('--- Calc Angle from Bearing')
        bearing = turn_angle
        return bearing
    
    def calc_bearing_from_points(self, x1, y1, x2, y2):
        '''
        Calculate compass bearing from (x1,y1) to (x2,y2)
        '''
        log.debug('--- Calc Bearing from Points')
        dx = x2 - x1
        dy = y2 - y1 
        angle_radians = np.arctan2(dx, dy) 
        angle_radians = np.mod(angle_radians + np.pi, 2*np.pi) - np.pi 
        return angle_radians

    def calc_next_point(self, xy_resolution, image_bounds): 
        '''
        When given a distance and angle calculate the X and Y coords of it
        when starting from a current position. 
        ''' 
        self.prev_x = self.current_x
        self.prev_y = self.current_y
        prev_x_proj, prev_y_proj = self.repast_to_proj(self.prev_x, self.prev_y, xy_resolution, image_bounds) 

        current_x_proj = prev_x_proj + self.step_distance*np.sin(self.heading_from_prev + self.turn_angle)*30.0
        current_y_proj = prev_y_proj + self.step_distance*np.cos(self.heading_from_prev + self.turn_angle)*30.0
        self.current_x, self.current_y = self.proj_to_repast(current_x_proj, current_y_proj, xy_resolution, image_bounds)
 
        x_centroid_proj, y_centroid_proj = self.repast_to_proj(self.centroid_x, self.centroid_y, xy_resolution, image_bounds)
    
        self.heading_to_centroid = self.calc_bearing_from_points(current_x_proj, current_y_proj, x_centroid_proj, y_centroid_proj)   
        self.heading_from_prev   = self.calc_bearing_from_points(prev_x_proj, prev_y_proj, current_x_proj, current_y_proj) 

        a = np.array((self.current_x, self.current_y))
        b = np.array((self.centroid_x, self.centroid_y))
        self.distance_to_centroid = np.linalg.norm(a-b) # Euclidian distance from current location to centroid. NOTE: this is in repast grid units not meters.

        return 
 