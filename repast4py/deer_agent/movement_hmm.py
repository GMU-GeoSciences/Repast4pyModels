import math 
from scipy.stats import exponweib, wrapcauchy, weibull_min
import numpy as np
import random 

from .movement_basic import BaseMoveModel
from .behaviour import Behaviour_State
 
import logging as pylog # Repast logger is called as "logging"
log = pylog.getLogger(__name__)
 
class HMM_MoveModel_2_States(BaseMoveModel):
    '''
    Simple random movement model. Agents move around with a weibull/cauchy step and turn model.
    Not influenced by environmental, time or behaviour states. Just a pure random walk. 
    '''
    def __init__(self, *args, **kwargs):
        self.movement_n_states = 2
        self.movement_params = [{'state': 0, 
                                 'state_name': 'HMM State 0',
                                 'step_params':{'c': 1.38666,
                                                'loc': 1,
                                                'scale': 29.142},
                                 'turn_params':{'c': 0.23,
                                                'loc': -3.123,
                                                'scale': 1}},
                                {'state': 1, 
                                 'state_name': 'HMM State 1',
                                 'step_params':{'c': 1.0378,
                                                'loc': 1,
                                                'scale': 148},
                                 'turn_params':{'c': 0.04,
                                                'loc': 0.1,
                                                'scale': 1}}
                                                ] # Params for each different move state. List of N dictionaries of step and turn params
        
        assert len(self.movement_params) == self.movement_n_states, "Too few movement params for number of movement states."

        # HMM Covariate Intercept matrix
        self.hmm_covariate_intercept = np.asarray([-8.359713e-01, -2.08067466])
        # HMM Covariate Raster Coeff matrix
        self.hmm_covariate_coeff = np.asarray([5.790579e-05,  0.01360621])

        assert len(self.hmm_covariate_intercept) == self.movement_n_states, "Covariate intercept matrix wrong size for number of hidden states."
        assert len(self.hmm_covariate_coeff) == self.movement_n_states, "Covariate coefficient matrix wrong size for number of hidden states."

        # TODO: HMM should randomly select the starting state based off the model fit params.
        self.current_state = 0
        self.next_state = 0
        self.step_resolution = 30.0


    def choose_next_state(self, current_state, location_array):
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
        # center_cover = location_array
        center_cover = np.take(location_array, location_array.size // 2)
        self.current_state = current_state

        B = self.hmm_covariate_intercept
        C = self.hmm_covariate_coeff
        
        yy = np.asarray([center_cover])*C+B
        y = np.exp(yy) / (1 + np.exp(yy))

        # y[0] == Prob of changing state from 0 to 1
        # y[1] == Prob of changing state from 1 to 0

        state_number = self.current_state.value

        if state_number == 0:
            if random.uniform(0, 1) < y[0]: 
                next_state = Behaviour_State(1)
            else:
                next_state = Behaviour_State(0)
        elif state_number == 1: 
            if random.uniform(0, 1) < y[1]: 
                next_state = Behaviour_State(0)
            else:
                next_state = Behaviour_State(1)
        else: 
            log.warning(f'Unknown state {current_state}')

        return next_state
    

    def calculate_random_step(self):
        '''
        Calculat the step distance based off a distribution function and the associated parameters.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
        ''' 
        state_number = self.current_state.value
        params = self.movement_params[state_number]['step_params']
        # assert self.movement_params[state_number]['state'] == self.current_state, "Bad state number..."
        step_distance = weibull_min.rvs(params['c'],    
                                  loc = params['loc'], 
                                  scale = params['scale'])
         
        return step_distance/self.step_resolution
    
    def calculate_random_turn(self):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html
        c == rho
        loc == mu
        '''
        state_number = self.current_state.value
        params = self.movement_params[state_number]['turn_params']
        turn_angle = wrapcauchy.rvs(params['c'], 
                                    loc = params['loc'],
                                    scale = params['scale']) 
        

        turn_angle = np.mod(turn_angle + np.pi, 2*np.pi) - np.pi
        return turn_angle
    

    def step(self, agent, location_array):
        '''
        For each timestep the agent needs to calculate a new position based off of some 
        environmental variables and the previous step/state.
        In this simple model location info doesn't matter and the agent doesn't move
        '''  

        # next_position, next_state = move_model.step(agent, local_cover, self.xy_resolution)
        current_state = agent.behaviour_state
        self.current_state = self.choose_next_state(current_state, location_array)
        step_distance = self.calculate_random_step()
        turn_angle = self.calculate_random_turn() 
        # self.current_state = next_state

        return self.current_state, step_distance, turn_angle