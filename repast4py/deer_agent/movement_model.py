import math 
from scipy.stats import exponweib, wrapcauchy
import numpy as np
from dataclasses import dataclass
from enum import Enum
import datetime

from time_functions import *
from behaviour import *

'''
This models the deer's movement based on a Hidden Markov Model derived from GPS data. 

The model assumes that there are some number of "Hidden States" that have an influence on the step distance and 
turning angle. It attempts to calculate the Hidden States and to fit Step and Turn distributions to them. It also can provide 
state transistion probabilites as a function of "covariates", which in this case can be the suitability raster file.

Read more here: 
https://cloud.r-project.org/web/packages/moveHMM/vignettes/moveHMM-guide.pdf
and
https://cran.r-project.org/web/packages/moveHMM/vignettes/moveHMM-example.pdf

It would be nice to rewrite the R package using Python libraries but that's a little out of scope for here. 
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

# ValueError: mutable default <class 'deer_agent.movement.Position_Vector'> for field pos is not allowed: use default_factory
# @dataclass
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
    step_distance: float = 0.0
    turn_angle: float = 0.0

    # def __post_init__(self):
    #     self.calc_dist_and_angle()

    # def calc_dist_and_angle(self):
        
    #     '''
    #     Calculate the distance and angle to centroid
    #     using euclidean maths. Great circle be damned!
    #     https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    #     https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points

    #     Need to do some fiddling to get North as 0 degrees
    #     '''
    #     a = np.array((self.current_point.x, self.current_point.y, self.current_point.z))
    #     b = np.array((self.centroid.x, self.centroid.y, self.centroid.z))
    #     self.distance_to_centroid = np.linalg.norm(a-b)
        
    #     dx = self.centroid.x - self.current_point.x
    #     dy = self.centroid.y - self.current_point.y
    #     angle_radians = np.arctan2(dx, dy)
    #     self.heading_to_centroid = (angle_radians) % (2 * np.pi) # Compass direction of travel between current_pos and centroid 

    #     dx = self.current_point.x - self.last_point.x
    #     dy = self.current_point.y - self.last_point.y
    #     angle_radians = np.arctan2(dx, dy)
    #     self.heading_from_prev = (angle_radians) % (2 * np.pi) # Compass direction of travel between last_pos and current_pos 
    #     return
    
    # def calc_next_point(self, initial_point, step_distance, turn_angle):
    #     '''
    #     When given a distance and angle calculate the X and Y coords of it
    #     when starting from a current position. 

    #     Turn Angle = Angle(Prev, Current) - Angle(Current, Next)
    #     '''
    #     self.calc_dist_and_angle()

    #     next_x = initial_point.x + step_distance*np.sin(turn_angle)
    #     next_y = initial_point.y + step_distance*np.cos(turn_angle)
        
    #     next_point = Point(next_x,next_y)
    #     return next_point

# def step(agent, xy_resolution):
#     '''
#     When given a distance and angle calculate the X and Y coords of it
#     when starting from a current position. 
#     '''
#     step_params = choose_params(agent.timestamp)
#     #TODO: Handle edge case of where xy_resolution[0] != xy_resolution[1]
#     agent.pos.step_distance = calculate_random_step(step_params)/ xy_resolution[0] 
#     agent.pos.step_angle = calculate_random_turn(agent) 
    
#     # Update the distances and angles:
#     # next_position = agent.pos.calc_next_point(agent.pos.current_point, agent.pos.step_distance, agent.pos.step_angle) 

#     return next_position

def choose_params(timestamp):
    this_season = check_time_of_year(timestamp)
    if this_season == DeerSeasons.GESTATION:
        step_params = StepTurnDist.GESTATION
    elif this_season == DeerSeasons.FAWNING:
        step_params = StepTurnDist.FAWNING
    elif this_season == DeerSeasons.PRERUT:
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

    if agent.behaviour_state == Behaviour_State.NORMAL:
        u_t = agent.pos.heading_to_centroid
        p_t = 0.5
        
    elif agent.behaviour_state == Behaviour_State.DISPERSE:
        u_t = agent.pos.heading_from_prev
        p_t = 0.5

    elif agent.behaviour_state == Behaviour_State.MATING:
        u_t = agent.pos.heading_to_centroid
        p_t = 0.5

    elif agent.behaviour_state == Behaviour_State.EXPLORE:
        u_t = agent.pos.heading_from_prev
        p_t = 0.5
    
    turn_angle = wrapcauchy.rvs(p_t, loc = u_t, scale = 1) 
    return turn_angle