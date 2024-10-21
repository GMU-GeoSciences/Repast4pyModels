from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
from typing import Tuple

import numpy as np

import logging as pylog # Repast logger is called as "logging"

log = pylog.getLogger(__name__)

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

        # Infection Vars:
        self.is_infected = False # Is infected with Cov19 virus
        self.infected_duration = 0 # How many days/hours/steps this agent has been infected
        
        # Deer Vars
        self.home_range_centroid = (0,0) #Home range centroid x,y. 
        self.home_range_diameter = 2 # Diameter of home range, assuming it's all circular. 
        self.speed_scaling_factor = 1 # Grid scaling factor. 1 == deer moves ~1 spatial unit per tick
        self.behaviour_state = 0 #Some indicator of behaviour; foraging, resting, travelling etc. Can maybe be a tuple?
        
        # Agent Control Vars
        self.random_starter_seed = 12345 # A psuedo random number generator seed. (Hopefully) allows repeat performances. 
        ## Possible other vars? 
        # self.age
        # self.is_male
        # self.has_child

    def save(self) -> Tuple:
        """Saves the state of this agent as a Tuple.

        Used to move this agent from one MPI rank to another.

        Returns:
            The saved state of this agent.

        Example: ((5, 0, 1), False, 0, (0, 0), 2, 1, 0)
        """ 
        return (self.uid, 
                self.is_infected, 
                self.infected_duration, 
                self.home_range_centroid, 
                self.home_range_diameter,
                self.speed_scaling_factor,
                self.behaviour_state)

    def expose_to_infection(self, tick):
        '''
        Deer is exposed to infection. State changes depend on immunity, if already sick, etc 
        '''
        self.is_infected = True
        self.infected_duration = 0 
        # TODO: This needs to be more complex
        return

    def calculate_next_pos(self, last_x, last_y, tick):
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
        # TODO: This needs to be more complex
        this_seed = int(self.random_starter_seed + tick) # Increment seed on each tick.
        centroid_x = self.home_range_centroid[0]
        centroid_y = self.home_range_centroid[1]
        home_range_d = self.home_range_diameter
        
        rng = np.random.default_rng(this_seed)
        x_delta, y_delta = rng.standard_normal(2)*self.speed_scaling_factor

        if centroid_x - last_x > home_range_d:
            x_delta = abs(x_delta) # Make next X component towards centroid
        elif last_x - centroid_x > home_range_d:
            x_delta = 0 - abs(x_delta) # Make next X component towards centroid

        if centroid_y - last_y > home_range_d:
            y_delta = abs(y_delta) # Make next X component towards centroid
        elif last_y - centroid_y > home_range_d:
            y_delta = 0 - abs(y_delta) # Make next X component towards centroid


        if abs(centroid_x - last_x) > home_range_d:
            # The last known X is too far away from centroid
            x_delta = 0 - abs(x_delta) # Make next X component towards centroid
        if abs(centroid_y - last_y) > home_range_d:
            # The last known Y is too far away from centroid
            x_delta = 0 - abs(x_delta) # Make next Y component towards centroid

        next_x = last_x + x_delta
        next_y = last_y + y_delta

        return (next_x, next_y)
    
    def calculate_next_state(self):
        '''
        Take current time of day, previous behaviour(s?), and location. 
        Use these to determine the behaviour of the deer. 

         Seeing as how this step is going to be run billions of times 
         computational efficiency is important. 
        '''
        # TODO: This needs to be more complex
        self.behaviour_state = 0
        self.infected_duration += 1 
        return

    def step(self, last_location, tick):
        '''
        Function for the agent class to change behaviour state, and location. 

        Should be called in the model.step function. 
        ''' 
        next_x, next_y = self.calculate_next_pos(last_location.x, last_location.y, tick)
        self.calculate_next_state()
        return next_x, next_y