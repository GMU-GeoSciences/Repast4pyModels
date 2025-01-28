import math 
from scipy.stats import exponweib, wrapcauchy
import numpy as np
from dataclasses import dataclass
from enum import Enum
import datetime

from . import time_functions
from . import behaviour

'''
This controls the stepping distances and angles
for different cases.


    DLD states a FOCUS wrapped cauchy distribution works best
    where the two cauchy params are defined by  
    μt = θct
    ρt = ρ∞ + (ρ0 – ρ∞)*exp(-γρ*dt)

    and

    θct= angle to turn directly to centroid
    dt = distance to centroid
    γρ = parameter controlling rate of convergence between ρ0 and ρ∞
    ρ0 = mean cosine of turn angles at the center of home range, 
    ρ∞ = mean cosine far from center of home range

'''

# Time Based Movement Parameters
# Weibull Distribution Params:
# Gestation

# From GPS Data
#           a	         c	         loc	    scale
# season				
# Fawning	5.264978	0.425295	-0.016784	7.518848
# Gestation	7.441196	0.357561	-0.008258	3.716973
# PreRut	4.478378	0.429409	-0.007190	10.923116
# Rut	    5.354084	0.383384	0.101771	7.730135

@dataclass
class StepTurnDist_mixin:
    '''
    Hold the step and turn distribution variables
    for randomly moving the agents around
    '''
    dist_name: str
    a: float
    c: float
    loc: float
    scale: float

class StepTurnDist(StepTurnDist_mixin, Enum):
    '''
    These units are in meters per hour. so need to get scaled when NOT
    using 1 hour time steps... But how? 
    '''
    GESTATION= 'Gestation', 7.441196, 0.357561, -0.008258, 3.716973
    FAWNING= 'Fawning', 5.264978, 0.425295, -0.016784, 7.518848
    PRERUT='PreRut', 4.478378, 0.429409, -0.007190, 10.923116
    RUT='Rut', 5.354084, 0.383384, 0.101771, 7.730135

@dataclass
class Point:
    '''
    Hold X,Y and optionally Z coords.
    '''
    x: float
    y: float   
    z: float = 0.0 
    
@dataclass
class Position_Vector:
    last_point: Point
    current_point: Point
    centroid: Point
    heading_to_centroid: float = 0.0
    distance_to_centroid: float = 0.0
    heading_from_prev: float = 0.0

    def __post_init__(self):
        self.calc_dist_and_angle()

    def calc_dist_and_angle(self):
        
        '''
        Calculate the distance and angle to centroid
        using euclidean maths. Great circle be damned!
        https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points

        Need to do some fiddling to get North as 0 degrees
        '''
        a = np.array((self.current_point.x, self.current_point.y, self.current_point.z))
        b = np.array((self.centroid.x, self.centroid.y, self.centroid.z))
        self.distance_to_centroid = np.linalg.norm(a-b)
        
        dx = self.centroid.x - self.current_point.x
        dy = self.centroid.y - self.current_point.y
        angle_radians = np.arctan2(dx, dy)
        self.heading_to_centroid = np.rad2deg((angle_radians) % (2 * np.pi)) # Compass direction of travel between current_pos and centroid 

        dx = self.current_point.x - self.last_point.x
        dy = self.current_point.y - self.last_point.y
        angle_radians = np.arctan2(dx, dy)
        self.heading_from_prev = np.rad2deg((angle_radians) % (2 * np.pi)) # Compass direction of travel between last_pos and current_pos 
    
    def calc_next_point(self, intial_point, step_distance, turn_angle):
        '''
        When given a distance and angle calculate the X and Y coords of it
        when starting from a current position. 

        Turn Angle = Angle(Prev, Current) - Angle(Current, Next)
        '''
        self.calc_dist_and_angle()

        next_x = intial_point.x + step_distance*np.sin(np.deg2rad(turn_angle))
        next_y = intial_point.y + step_distance*np.cos(np.deg2rad(turn_angle))
        
        next_point = Point(next_x,next_y)
        return next_point

def step(agent, xy_resolution):
    '''
    When given a distance and angle calculate the X and Y coords of it
    when starting from a current position. 
    '''
    step_params = choose_params(agent.timestamp)
    #TODO: Handle edge case of where xy_resolution[0] != xy_resolution[1]
    step_distance = calculate_random_step(step_params)/ xy_resolution[0] 
    step_angle = calculate_random_turn(agent)
    current_pos = agent.pos.current_point
    
    # Update the distances and angles:
    next_position = agent.pos.calc_next_point(current_pos, step_distance, step_angle) 

    return next_position

def choose_params(timestamp):
    this_season = time_functions.check_time_of_year(timestamp)
    if this_season == time_functions.DeerSeasons.GESTATION:
        step_params = StepTurnDist.GESTATION
    elif this_season == time_functions.DeerSeasons.FAWNING:
        step_params = StepTurnDist.FAWNING
    elif this_season == time_functions.DeerSeasons.PRERUT:
        step_params = StepTurnDist.PRERUT
    else: 
        step_params = StepTurnDist.RUT

    return step_params

def calculate_random_step(step_params):
    '''
    Takes timestamp, extracts month/hour from it and chooses correct
    mode of stepping.  
    '''  
    # TODO: How to scale the distance depending on tick step size.
    step_distance = exponweib.rvs(step_params.a, 
                                  step_params.c, 
                                  step_params.loc, 
                                  step_params.scale)

    return step_distance

def calculate_random_turn(agent):
    '''
    This calculates a random step by creating a random distribution
    using the distance and turn angle to the centroid. 

    This is an implementation of the "simple return" distribution from DLD paper.

    for stats package:
        c = rho_t
        x = u_t 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html#scipy.stats.wrapcauchy
    ''' 
    # TODO: Not using the distance from centroid?
    distance_from_centroid = agent.pos.distance_to_centroid

    if agent.behaviour_state == behaviour.Behaviour_State.NORMAL:
        u_t = np.deg2rad(agent.pos.heading_to_centroid)
        p_t = 0.5
        
    elif agent.behaviour_state == behaviour.Behaviour_State.DISPERSE:
        u_t = np.deg2rad(agent.pos.heading_from_prev)
        p_t = 0.5

    elif agent.behaviour_state == behaviour.Behaviour_State.MATING:
        u_t = np.deg2rad(agent.pos.heading_to_centroid)
        p_t = 0.5

    elif agent.behaviour_state == behaviour.Behaviour_State.EXPLORE:
        u_t = np.deg2rad(agent.pos.heading_from_prev)
        p_t = 0.5
    
    turn_angle = wrapcauchy.rvs(p_t, loc = u_t, scale = 1) 
    return np.rad2deg(turn_angle)