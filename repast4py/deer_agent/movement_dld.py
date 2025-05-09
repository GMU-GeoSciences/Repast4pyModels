from scipy.stats import exponweib, wrapcauchy, weibull_min
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .movement_basic import BaseMoveModel
from .time_functions import * 
from .behaviour import *
 
import logging as pylog # Repast logger is called as "logging"
log = pylog.getLogger(__name__)

# ValueError: mutable default <class 'deer_agent.movement.Position_Vector'> for field pos is not allowed: use default_factory
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


class DLD_MoveModel(BaseMoveModel):
    '''
    Simple random movement model. Agents move around with a weibull/cauchy step and turn model.
    Not influenced by environmental, time or behaviour states. Just a pure random walk. 
    '''
    def __init__(self, *args, **kwargs): 
        pass
 
    def calculate_random_step(self, agent):
        '''
        Calculate the step distance based off a distribution function and the associated parameters.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
        '''
        this_season = check_time_of_year(agent.timestamp)

        if this_season == DeerSeasons.GESTATION:
            step_params = StepTurnDist.GESTATION
        elif this_season == DeerSeasons.FAWNING:
            step_params = StepTurnDist.FAWNING
        elif this_season == DeerSeasons.PRERUT:
            step_params = StepTurnDist.PRERUT
        else: 
            step_params = StepTurnDist.RUT 
 
        step_distance = exponweib.rvs(step_params.a, 
                                    step_params.c, 
                                    step_params.loc, 
                                    step_params.scale)
        return step_distance
    
    def calculate_random_turn(self, agent):
        '''
        This calculates a random step by creating a random distribution
        using the distance and turn angle to the centroid. 
        ''' 
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
        
        turn_angle = wrapcauchy.rvs(p_t, 
                                    loc = u_t, 
                                    scale = 1)  
        
        turn_angle = np.mod(turn_angle + np.pi, 2*np.pi) - np.pi

        return turn_angle 

    def step(self, agent):
        '''
        For each timestep the agent needs to calculate a new position based off of some 
        environmental variables and the previous step/state.
        In this simple model location info doesn't matter and the agent doesn't move
        Behaviour_state and TimeState are held in the agent object 
        '''
        step_distance = self.calculate_random_step(agent)
        turn_angle = self.calculate_random_turn(agent)  

        return step_distance, turn_angle