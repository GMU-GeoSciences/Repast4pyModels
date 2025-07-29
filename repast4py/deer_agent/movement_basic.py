import math 
from scipy.stats import exponweib, wrapcauchy, weibull_min
import numpy as np
import random 
from dataclasses import dataclass
from enum import Enum
 
import logging as pylog # Repast logger is called as "logging"
log = pylog.getLogger(__name__)

class BaseMoveModel(object): 
    '''
    Basic movement model. This should be overloaded in more complex movement models.
    '''

    def __init__(self): 
        self.movement_n_states = 1 # Number of different states in the movement model.
        self.movement_params = [{}] # Params for each different move state. List of N dictionaries 
        assert len(self.movement_params) == self.movement_n_states, "Too few movement params for number of movement states."

        self.current_state = 0
        self.next_state = 0
        self.step_resolution = 30 #Required to go from distribution params to xy grid dimensions...
        return

    # functions:  
    def choose_next_state(self, location_info = {}):
        '''
        This function uses the location_info dict to decide
        whether to change the current movement state.
        '''
        next_state = self.current_state
        return next_state
    
    def calculate_random_step(self):
        '''
        Calculate a random distance to step
        '''
        step_distance = 1
        return step_distance
    
    def calculate_random_turn(self):
        '''
        Go straight, or 90 deg left. (why is positive turn left?) 
        '''
        turn_angle = np.random.choice([-np.pi/2, 0 , np.pi/2])
        return turn_angle
    
    def step(self):
        '''
        For each timestep the agent needs to calculate a new position based off of some 
        environmental variables and the previous step/state.
        In this simple model location info doesn't matter and the agent doesn't move
        '''  
        self.next_state = self.choose_next_state()
        step_distance = self.calculate_random_step()
        turn_angle = self.calculate_random_turn() 

        # Update movement model for next step.
        self.current_state = self.next_state

        return step_distance, turn_angle


class RandomMovement(BaseMoveModel):
    '''
    Simple random movement model. Agents move around with a weibull/cauchy step and turn model.
    Not influenced by environmental, time or behaviour states. Just a pure random walk. 
    ''' 
    def __init__(self, *args, **kwargs):
        self.movement_n_states = 1
        self.step_resolution = 30 #Required to go from distribution params to xy grid dimensions... 
        self.movement_params = [{'state': 0, 
                                 'state_name': None,
                                 'step_params':{'a':7.44,
                                                'c': 0.35,
                                                'loc': -0.08,
                                                'scale': 3.71},
                                 'turn_params':{'c': 0.23,
                                                'loc': -3.14,
                                                'scale': 1}}] # Params for each different move state. List of N dictionaries of step and turn params
        
        assert len(self.movement_params) == self.movement_n_states, "Too few movement params for number of movement states."
        self.current_state = 0
        self.next_state = 0

    def calculate_random_step(self):
        '''
        Calculat the step distance based off a distribution function and the associated parameters.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
        ''' 
        params = self.movement_params[self.next_state]['step_params']
        assert self.movement_params[self.next_state]['state'] == self.next_state, "Bad state number..."
        # step_distance = weibull_min.rvs(params['c'],    
        #                           loc = params['loc'], 
        #                           scale = params['scale']) 
        step_distance = exponweib.rvs(params['a'], 
                                    params['c'], 
                                    params['loc'], 
                                    params['scale'])
        
        return step_distance/self.step_resolution
    
    def calculate_random_turn(self):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html
        c == rho
        loc == mu
        '''
        params = self.movement_params[self.next_state]['turn_params']
        turn_angle = wrapcauchy.rvs(params['c'], 
                                    loc = params['loc'],
                                    scale = params['scale'])   

        
        turn_angle = np.mod(turn_angle + np.pi, 2*np.pi) - np.pi
        return turn_angle
    
####################################
## Movement Helper Classes        ##
####################################
# @dataclass
# class Point:
#     '''
#     Hold X,Y and optionally Z coords.
#     '''
#     x: float
#     y: float   
#     z: float = 0.0 
    

# @dataclass
# class Position_Vector:
#     last_point: Point
#     current_point: Point
#     centroid: Point
#     heading_to_centroid: float = 0.0
#     distance_to_centroid: float = 0.0
#     heading_from_prev: float = 0.0
#     step_distance: float = 0.0
#     turn_angle: float = 0.0

#     def __post_init__(self):
#         self.calc_dist_and_angle()

#     def calc_dist_and_angle(self):
        
#         '''
#         Calculate the distance and angle to centroid
#         using euclidean maths. Great circle be damned!
#         https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
#         https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points

#         Need to do some fiddling to get North as 0 degrees
#         '''
#         a = np.array((self.current_point.x, self.current_point.y, self.current_point.z))
#         b = np.array((self.centroid.x, self.centroid.y, self.centroid.z))
#         self.distance_to_centroid = np.linalg.norm(a-b)
        
#         dx = self.centroid.x - self.current_point.x
#         dy = self.centroid.y - self.current_point.y
#         angle_radians = np.arctan2(dx, dy)
#         self.heading_to_centroid = (angle_radians) % (2 * np.pi) # Compass direction of travel between current_pos and centroid 

#         dx = self.current_point.x - self.last_point.x
#         dy = self.current_point.y - self.last_point.y
#         angle_radians = np.arctan2(dx, dy)
#         self.heading_from_prev = (angle_radians) % (2 * np.pi) # Compass direction of travel between last_pos and current_pos 
    
#     def calc_next_point(self, initial_point, step_distance, turn_angle):
#         '''
#         When given a distance and angle calculate the X and Y coords of it
#         when starting from a current position. 

#         Turn Angle = Angle(Prev, Current) - Angle(Current, Next)
#         '''
#         self.calc_dist_and_angle()

#         next_x = initial_point.x + step_distance*np.si(turn_angle)
#         next_y = initial_point.y + step_distance*np.cos(turn_angle)
        
#         next_point = Point(next_x,next_y)
#         return next_point
    

# @dataclass
# class StepTurnDist_mixin:
#     '''
#     Hold the step and turn distribution variables
#     for randomly moving the agents around
#     '''
#     dist_name: str
#     a: float
#     c: float
#     loc: float
#     scale: float

# class StepTurnDist(StepTurnDist_mixin, Enum):
#     '''
#     These units are in meters per hour. so need to get scaled when NOT
#     using 1 hour time steps... But how? 
#     '''
#     GESTATION= 'Gestation', 7.441196, 0.357561, -0.008258, 3.716973
#     FAWNING= 'Fawning', 5.264978, 0.425295, -0.016784, 7.518848
#     PRERUT='PreRut', 4.478378, 0.429409, -0.007190, 10.923116
#     RUT='Rut', 5.354084, 0.383384, 0.101771, 7.730135