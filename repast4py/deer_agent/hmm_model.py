import math 
from scipy.stats import exponweib, wrapcauchy, weibull_min
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum
import datetime
 
import logging as pylog # Repast logger is called as "logging"

log = pylog.getLogger(__name__)

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
 
# class Behaviour_State(Enum): 
#     NORMAL = 1
#     DISPERSE = 2
#     MATING = 3
#     EXPLORE = 4


# class DeerSeasons(TimeOfYear_mixin, Enum):
#     ''' 
#     GESTATION = 'Gestation', 1,1,14,5 
#     FAWNING =   'Fawning',  15,5,31,8 
#     PRERUT =    'PreRut',    1,9,31,10 
#     RUT =       'Rut',       1,11,31,12 

class BehaviourState_HMM():

    # params:
    def __init__(self):
        # Defaults
        self.n_states = 2 
        return

    # functions:  
    def choose_next_state(self, behaviour_state, local_cover):
        '''
        Based off the: 
            - current state, 
            - the location suitability, 
            - and the transition matrix,
        Randomly choose the next state. 

        Behaviour state is from the behaviour.py values and the probability calculated in the R HMM model.

        Regression coeffs for the transition probabilities:
        Is this for a logloss regression model? Logit? 
        --------------------------------------------------
                            1 -> 2      2 -> 1
        intercept    -7.871303e-01 -2.08200610
        raster_value  9.414977e-05  0.01350731
        TODO: Rewrite this for N behaviour states
        '''
        center_cover = np.take(local_cover, local_cover.size // 2)

        B = np.asarray([-8.359713e-01, -2.08067466]) # Intercepts
        C = np.asarray([5.790579e-05,  0.01360621])  # Raster Coeff
        
        yy = np.asarray([center_cover])*C+B
        y = np.exp(yy) / (1 + np.exp(yy))

        # Y[0] == Prob of changing state from 0 to 1
        # Y[1] == Prob of changing state from 1 to 0
        if behaviour_state == Behaviour_State.NORMAL:
            if random.uniform(0, 1) < y[0]: 
                next_state = Behaviour_State.EXPLORE
            else:
                next_state = Behaviour_State.NORMAL
        else: 
            if random.uniform(0, 1) < y[1]: 
                next_state = Behaviour_State.NORMAL
            else:
                next_state = Behaviour_State.EXPLORE
        return next_state
    
    def choose_params(self, this_season, agent_is_male, behaviour_state):
        '''
        From the Time of Year, and the Agent's Gender, pick some 
        step and turn PDF parameters.

        HARD CODED FROM R HMM! NOT GREAT!

            Value of the maximum log-likelihood: -1733616 

            Step length parameters:
            ----------------------
                        state 1      state 2
            shape     1.363459e+00 1.037382e+00
            scale     2.914216e+01 1.477976e+02
            zero-mass 3.863936e-05 2.156298e-05

            Turning angle parameters:
            ------------------------
                            state 1   state 2
            mean          -3.123260 0.1007629
            concentration  0.230288 0.0405042

            Regression coeffs for the transition probabilities:
            --------------------------------------------------
                                1 -> 2      2 -> 1
            intercept    -8.359713e-01 -2.08067466
            raster_value  5.790579e-05  0.01360621

            Initial distribution:
            --------------------
            [1] 0.3943988 0.6056012

        '''
        # log.info(f'Agent is male? {agent_is_male}. Season: {this_season}. State: {behaviour_state}')
        if agent_is_male:
            # Males 
            if this_season == DeerSeasons.GESTATION:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
                    
            elif this_season == DeerSeasons.FAWNING:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
            elif this_season == DeerSeasons.PRERUT:
                if behaviour_state == 0:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
            elif this_season == DeerSeasons.RUT:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
        else:
            # Females
            if this_season == DeerSeasons.GESTATION:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
                    
            elif this_season == DeerSeasons.FAWNING:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
            elif this_season == DeerSeasons.PRERUT:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}
            elif this_season == DeerSeasons.RUT:
                if behaviour_state == Behaviour_State.NORMAL:
                    step_params = {"c":1.38666,
                                   "loc":1,
                                   "scale":29.142}
                    turn_params = {"c":0.23,
                                   "loc":-3.123,
                                   "scale":1}
                
                elif behaviour_state == Behaviour_State.EXPLORE:
                    step_params = {"c":1.0378,
                                   "loc":1,
                                   "scale":147.7976}
                    turn_params = {"c":0.04,
                                   "loc":0.1,
                                   "scale":1}

        return step_params, turn_params
    
    def calculate_random_step(self, params, heading_to_centroid = None, use_centroid = False):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponweib.html#scipy.stats.exponweib
        ''' 
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
        step_distance = weibull_min.rvs(params['c'],
                                  loc = params['loc'], 
                                  scale = params['scale']) 
        return step_distance
    
    def calculate_random_turn(self, params):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html
        c == rho
        loc == mu
        '''
        turn_angle = wrapcauchy.rvs(params['c'], 
                                    # loc = params['loc'], # this LOC thing really screws up the distribution...
                                    scale = params['scale']) 

        return turn_angle
    
    def step(self, agent, local_cover, xy_resolution, use_centroid = False):
        '''
        When given a distance and angle calculate the X and Y coords of it
        when starting from a current position. 
        ''' 
        this_season = check_time_of_year(agent.timestamp)

        next_state = self.choose_next_state(agent.behaviour_state, local_cover)  
        step_params,turn_params = self.choose_params(this_season, agent.is_male, next_state)  
        agent.pos.step_distance = self.calculate_random_step(step_params) / xy_resolution[0]
        agent.pos.turn_angle = self.calculate_random_turn(turn_params)

        # if use_centroid:
            # TODO: How to implement the centroid in the turning angle? Which one is positive? 
            # step_angle = agent.pos.heading_to_centroid - step_angle 
        current_pos = agent.pos.current_point
        # Update the distances and angles:
        next_position = agent.pos.calc_next_point(current_pos, agent.pos.step_distance, agent.pos.turn_angle) 

        return next_position, next_state