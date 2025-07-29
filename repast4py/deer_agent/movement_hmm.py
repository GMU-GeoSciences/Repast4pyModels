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
    2 State HMM model. Takes suitability raster data into account
    '''
    def __init__(self, *args, **kwargs):
        self.movement_n_states = 2
        self.movement_params = [{'state': 0, 
                                 'state_name': 'HMM State 0',
                                 'step_params':{'c': 1.042846e+00,
                                                'loc': 1,
                                                'scale': 1.471591e+02},
                                 'turn_params':{'c': 0.0396948,
                                                'loc': -0.1339658,
                                                'scale': 1}},
                                {'state': 1, 
                                 'state_name': 'HMM State 1',
                                 'step_params':{'c': 1.356711e+00,
                                                'loc': 1,
                                                'scale': 2.911879e+01},
                                 'turn_params':{'c': 0.2290252,
                                                'loc': 3.1166163,
                                                'scale': 1}}
                                                ] # Params for each different move state. List of N dictionaries of step and turn params
        
        assert len(self.movement_params) == self.movement_n_states, "Too few movement params for number of movement states." 

        # HMM Covariate Intercept matrix
        self.hmm_covariate_intercept = np.asarray([-2.0696753, -0.777699801])
        # HMM Covariate Raster Coeff matrix
        self.hmm_covariate_coeff = np.asarray([0.1238317,  -0.005702054])

        assert len(self.hmm_covariate_intercept) == self.movement_n_states, "Covariate intercept matrix wrong size for number of hidden states."
        assert len(self.hmm_covariate_coeff) == self.movement_n_states, "Covariate coefficient matrix wrong size for number of hidden states."

        # TODO: HMM should randomly select the starting state based off the model fit params:
        # Initial distribution:
        # --------------------
        # [1] 0.5663704 0.4336296
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
    


class HMM_MoveModel_3_States_Canopy(BaseMoveModel):
    '''
    2 State HMM model. Takes canopy raster data into account
    '''
    def __init__(self, *args, **kwargs): 
        self.movement_n_states = 3
        self.home_range_radius = 1000 # [meters] or is this [pixels] ?
        self.movement_params = [{'state': 0, 
                                 'state_name': 'HMM State 0',
                                 'step_params':{'c':1.228546,
                                                'loc': 1,
                                                'scale': 1.133292e+02},
                                 'turn_params':{'c': 0.04079163,
                                                'loc': -3.08108420,
                                                'scale': 1}},
                                {'state': 1, 
                                 'state_name': 'HMM State 1',
                                 'step_params':{'c': 1.476773,
                                                'loc': 1,
                                                'scale': 2.405681e+01},
                                 'turn_params':{'c': 0.2053319,
                                                'loc': 3.1140039,
                                                'scale': 1}},
                                {'state': 2, 
                                 'state_name': 'HMM State 2',
                                 'step_params':{'c': 1.297833,
                                                'loc': 1,
                                                'scale': 3.318213e+02},
                                 'turn_params':{'c': 0.2165741,
                                                'loc': -0.0631689,
                                                'scale': 1}}
                                                ] # Params for each different move state. List of N dictionaries of step and turn params
        
        assert len(self.movement_params) == self.movement_n_states, "Too few movement params for number of movement states." 

        # HMM Covariate Intercept matrix
        self.hmm_covariate_intercept = np.asarray([-1.80329097,-2.332902740,-0.842190949,-2.07295724,-1.333573276,-1.95697222])
        # HMM Covariate Raster Coeff matrix
        self.hmm_covariate_coeff = np.asarray([0.01124452,-0.001542683,0.003304142,-0.01484277,0.005627201,0.01540365]) 
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
    
    def calculate_random_turn(self, size = 2):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html
        c == rho
        loc == mu
        '''
        state_number = self.current_state.value
        params = self.movement_params[state_number]['turn_params']
        turn_angle = wrapcauchy.rvs(params['c'], 
                                    loc = params['loc'],
                                    scale = params['scale'],
                                    size = size) 
        turn_angle = np.mod(turn_angle + np.pi, 2*np.pi) - np.pi
        return turn_angle
    

    def step(self, agent, location_array):
        '''
        For each timestep the agent needs to calculate a new position based off of some 
        environmental variables and the previous step/state.
        In this simple model location info doesn't matter and the agent doesn't move
        '''  

        # next_position, next_state = move_model.step(agent, local_cover, self.xy_resolution) 
        self.current_state = self.choose_next_state( agent.behaviour_state, location_array)
        step_distance = self.calculate_random_step()
        # turn_angles = self.calculate_random_turn() 
        ######
        # Pick a couple of turn angles, and choose the one closest to the return home angle
        # BUT only if distance from home range centroid > some_value
        ######
        return_home_angle = agent.heading_to_centroid - agent.heading_from_prev
        return_home_distance = agent.distance_to_centroid
        if return_home_distance > self.home_range_radius/self.step_resolution:
            # log.info('Too Far from HR Centroid!')
            turn_angles = self.calculate_random_turn(size=1) 
            turn_angle = turn_angles[(np.abs(turn_angles - return_home_angle)).argmin()]
        else:
            turn_angles = self.calculate_random_turn(size=1) 
            turn_angle = turn_angles[0]
        # self.current_state = next_state

        return self.current_state, step_distance, turn_angle
    

class HMM_MoveModel_3_States(BaseMoveModel):
    '''
    3 State HMM model. Takes raster data into account. This was trained using the suitability raster and not the canopy raster
    '''
    def __init__(self, *args, **kwargs):
        self.movement_n_states = 3
        self.home_range_limit = 1000 # [meters]?
        self.movement_params = [{'state': 0, 
                                 'state_name': 'HMM State 0',
                                 'step_params':{'c':1.311337,
                                                'loc': 1,
                                                'scale': 3.351919e+02},
                                 'turn_params':{'c': 0.21499672,
                                                'loc': -0.06124627,
                                                'scale': 1}},
                                {'state': 1, 
                                 'state_name': 'HMM State 1',
                                 'step_params':{'c': 1.231843,
                                                'loc': 1,
                                                'scale': 1.132833e+02},
                                 'turn_params':{'c': 0.04074727,
                                                'loc': -3.07373644,
                                                'scale': 1}},
                                {'state': 2, 
                                 'state_name': 'HMM State 2',
                                 'step_params':{'c': 1.470787,
                                                'loc': 1,
                                                'scale': 2.408109e+01},
                                 'turn_params':{'c': 0.2042132,
                                                'loc': 3.1127951,
                                                'scale': 1}}
                                                ] # Params for each different move state. List of N dictionaries of step and turn params
        
        assert len(self.movement_params) == self.movement_n_states, "Too few movement params for number of movement states." 

        # HMM Covariate Intercept matrix
        #                                               1 -> 2      1 -> 3      2 -> 1      2 -> 3     3 -> 1    3 -> 2
        self.hmm_covariate_intercept = np.asarray([-1.61019298,-1.62130884,-1.79254789,-1.72515496,-2.2150039,-0.665865790])
 
        # HMM Covariate Raster Coeff matrix
        self.hmm_covariate_coeff = np.asarray([0.09487647,  0.09543828, -0.08549171,  0.09086488, -0.1138435, 0.006435542])

        # assert len(self.hmm_covariate_intercept) == self.movement_n_states, "Covariate intercept matrix wrong size for number of hidden states."
        # assert len(self.hmm_covariate_coeff) == self.movement_n_states, "Covariate coefficient matrix wrong size for number of hidden states."

        # TODO: HMM should randomly select the starting state based off the model fit params.
        # Initial distribution:
        # --------------------
        # [1] 0.1190081 0.4928453 0.3881465
        self.current_state = 0 
        self.step_resolution = 30.0

    def get_transition_matrix(self, current_covariate):
            
        B = self.hmm_covariate_intercept
        C = self.hmm_covariate_coeff
        n_states = self.movement_n_states
        
        yy = np.asarray([current_covariate])*C+B
        y = np.exp(yy) / (1 + np.exp(yy))
        y_index = 0
        state_probs = np.zeros([n_states, n_states])
        for this_state in range(n_states):
            for next_state in range(n_states):
                if this_state == next_state:
                    #Skip!
                    continue
                else:
                    state_probs[this_state,next_state] = y[y_index] 
                y_index += 1
        
        for this_state in range(n_states):
            state_probs[this_state, this_state] = 1 - state_probs.sum(axis=1)[this_state]
        return state_probs
    
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

        prob_matrix = self.get_transition_matrix(center_cover)
        self.current_state = current_state
        state_number = self.current_state.value
        this_state_transistion = prob_matrix[state_number]
        next_state = np.random.choice(3, 1, p=this_state_transistion)
 
        return Behaviour_State(next_state)
    

    def calculate_random_step(self):
        '''
        Calculate the step distance based off a distribution function and the associated parameters.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min

        return it as a number of pixels...
        ''' 
        state_number = self.current_state.value
        params = self.movement_params[state_number]['step_params']
        # assert self.movement_params[state_number]['state'] == self.current_state, "Bad state number..."
        step_distance = weibull_min.rvs(params['c'],    
                                  loc = params['loc'], 
                                  scale = params['scale'])
         
        return step_distance/self.step_resolution
    
    def calculate_random_turn(self, size = 2):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html
        c == rho
        loc == mu
        '''
        state_number = self.current_state.value
        params = self.movement_params[state_number]['turn_params']
        turn_angle = wrapcauchy.rvs(params['c'], 
                                    loc = params['loc'],
                                    scale = params['scale'],
                                    size = size) 
        

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
        turn_angles = self.calculate_random_turn() 
        ######
        # Pick a couple of turn angles, and choose the one closest to the return home angle
        # BUT only if distance from home range centroid > some_value
        ######
        return_home_angle = agent.heading_to_centroid - agent.heading_from_prev
        return_home_distance = agent.distance_to_centroid
        if return_home_distance > self.home_range_limit/self.step_resolution:
            turn_angle = turn_angles[(np.abs(turn_angles - return_home_angle)).argmin()]
        else:
            turn_angle = turn_angles[0]
        # self.current_state = next_state

        return self.current_state, step_distance, turn_angle

class HMM_MoveModel_3_States_Canopy_Gender(BaseMoveModel):
    '''
    3 State HMM model that has been fit to male/female GPS data. 
    Takes canopy raster data into account instead of suitability raster.
    '''
    def __init__(self, *args, **kwargs): 
        self.movement_n_states = 3
        self.turn_choices = 2 # [meters]. This can be overridded on class init in the model.step()
        self.home_range_radius = 1000 # [meters]. This can be overridded on class init in the model.step()
        self.male_movement_params = [{'state': 0, 
                                    'state_name': 'HMM State 0',
                                    'step_params':{'c':1.212249,
                                                'loc': 1,
                                                'scale': 2.862531e+02},
                                 'turn_params':{'c': 0.23950750,
                                                'loc': -0.02439658,
                                                'scale': 1}},
                                {'state': 1, 
                                 'state_name': 'HMM State 1',
                                 'step_params':{'c': 1.271055e+00,
                                                'loc': 1,
                                                'scale': 8.971704e+01},
                                 'turn_params':{'c': 0.1106292,
                                                'loc': 3.1142667,
                                                'scale': 1}},
                                {'state': 2, 
                                 'state_name': 'HMM State 2',
                                 'step_params':{'c': 1.547475e+00,
                                                'loc': 1,
                                                'scale': 2.208336e+01},
                                 'turn_params':{'c': 0.2005346,
                                                'loc': 3.1173788,
                                                'scale': 1}}
                                                ] # Params for each different move state. List of N dictionaries of step and turn params
        
        self.female_movement_params = [{'state': 0, 
                                 'state_name': 'HMM State 0',
                                 'step_params':{'c':1.181699,
                                                'loc': 1,
                                                'scale': 1.327624e+02},
                                 'turn_params':{'c': 0.02428817,
                                                'loc': -2.88861242,
                                                'scale': 1}},
                                {'state': 1, 
                                 'state_name': 'HMM State 1',
                                 'step_params':{'c': 1.406792e+00,
                                                'loc': 1,
                                                'scale': 2.645763e+01},
                                 'turn_params':{'c': 0.2035553,
                                                'loc': 3.1114538,
                                                'scale': 1}},
                                {'state': 2, 
                                 'state_name': 'HMM State 2',
                                 'step_params':{'c': 1.084333e+00,
                                                'loc': 1,
                                                'scale':3.448261e+02},
                                 'turn_params':{'c': 0.41607627,
                                                'loc':-0.08637389,
                                                'scale': 1}}
                                                ] # Params for each different move state. List of N dictionaries of step and turn params
         
        # HMM Covariate Intercept matrix
        self.male_hmm_covariate_intercept = np.asarray([-2.43588239,-1.96748890,-1.045498445,-1.790791677,-1.3257993,-1.55521659])
        # HMM Covariate Raster Coeff matrix
        self.male_hmm_covariate_coeff = np.asarray([0.01689596,0.01480208,-0.005524766,0.009374563,-0.0128918, 0.01150192]) 
 
        self.female_hmm_covariate_intercept = np.asarray([-1.7812199,-3.8345379,-0.5354991864,-4.1782463,-0.4855525,-2.017667]) 
        self.female_hmm_covariate_coeff = np.asarray([0.0129289,0.0136312,-0.0007961524,-0.1103394,3.6702500, -1.986867]) 

        self.current_state = 0
        self.next_state = 0
        self.step_resolution = 30.0


    def choose_next_state(self, current_state, location_array, agent):
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
        if agent.is_male:
            B = self.male_hmm_covariate_intercept
            C = self.male_hmm_covariate_coeff
        else:
            B = self.female_hmm_covariate_intercept
            C = self.female_hmm_covariate_coeff
        
        yy = np.asarray([center_cover])*C+B
        y = np.exp(yy) / (1 + np.exp(yy)) 

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
    

    def calculate_random_step(self, agent):
        '''
        Calculat the step distance based off a distribution function and the associated parameters.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
        ''' 
        state_number = self.current_state.value
        if agent.is_male:
            params = self.male_movement_params[state_number]['step_params']
        else:
            params = self.female_movement_params[state_number]['step_params']
        # assert self.movement_params[state_number]['state'] == self.current_state, "Bad state number..."
        step_distance = weibull_min.rvs(params['c'],    
                                  loc = params['loc'], 
                                  scale = params['scale'])
         
        return step_distance/self.step_resolution
    
    def calculate_random_turn(self, agent, size = 2):
        '''
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html
        c == rho
        loc == mu
        '''
        state_number = self.current_state.value 
        if agent.is_male:
            params = self.male_movement_params[state_number]['turn_params']
        else:
            params = self.female_movement_params[state_number]['turn_params']

        turn_angle = wrapcauchy.rvs(params['c'], 
                                    loc = params['loc'],
                                    scale = params['scale'],
                                    size = size) 
        turn_angle = np.mod(turn_angle + np.pi, 2*np.pi) - np.pi
        return turn_angle
    
    def step(self, agent, location_array):
        '''
        For each timestep the agent needs to calculate a new position based off of some 
        environmental variables and the previous step/state.
        In this simple model location info doesn't matter and the agent doesn't move
        '''  

        # next_position, next_state = move_model.step(agent, local_cover, self.xy_resolution) 
        self.current_state = self.choose_next_state( agent.behaviour_state, location_array, agent)
        step_distance = self.calculate_random_step(agent)
        # turn_angles = self.calculate_random_turn() 
        ######
        # Pick a couple of turn angles, and choose the one closest to the return home angle
        # BUT only if distance from home range centroid > some_value
        ######
        # Must turn bearing/headings into turning angles.
        # turn_to_centroid = agent.heading_from_prev - agent.heading_to_centroid
        # return_home_turn = -agent.heading_to_centroid
        # return_home_turn = np.degrees(np.mod(np.radians(df_sim['heading_from_prev'] -  df_sim['turn_to_centroid']) +np.pi, 2*np.pi) -np.pi)

        return_home_turn = -np.mod(agent.heading_from_prev - agent.heading_to_centroid +np.pi, 2*np.pi) -np.pi
  
        return_home_distance = agent.distance_to_centroid
        if return_home_distance > self.home_range_radius/self.step_resolution: 
            turn_angles = self.calculate_random_turn(agent, size=self.turn_choices)
            choices = np.arctan2(np.sin(return_home_turn-turn_angles), np.cos(return_home_turn-turn_angles))
            turn_angle = turn_angles[(np.abs(choices)).argmin()]  
            
            # Force turn_home angle to be correct
            # turn_angle = -return_home_turn
            
        else:
            turn_angles = self.calculate_random_turn(agent, size=1) 
            turn_angle = turn_angles[0]
        # self.current_state = next_state

        return self.current_state, step_distance, turn_angle