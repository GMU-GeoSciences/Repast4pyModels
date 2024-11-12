from repast4py import core, random
from repast4py.space import DiscretePoint as dpt
from repast4py.space import ContinuousPoint as cpt
from typing import Tuple 

from datetime import datetime, timedelta
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
        self.is_infected = False    # Is infected with somthing
        self.is_contagious = False  # Is contagious to other agents 
        self.has_recovered = False  # Has recovered/immune
        self.disease_timer = 0      # How many days/hours/steps since this agent was infected/recovered
        
        # Agent Vars
        self.random_starter_seed = 12345 # A psuedo random number generator seed. (Hopefully) allows repeat performances. 
        self.group_id = None # Herd ID: used for grouping

        # Deer Vars
        self.birth_date = None # Birthday; useful for if we're going to run this model for years
        self.age = 0 # Age: for a quick check instead of calculating the age from birthdate every time
        self.is_male = False # Boolean gender value.
        self.is_fawn = False # Boolean for age classifier
        self.gestation_time = 0 # How long deer has been pregnant

        # Movement/Location Vars
        self.has_home_range = True # Agent has a home range.
        self.home_range_centroid = (0,0) #Home range centroid x,y.  
        self.direction_to_home_centroid = None # Bearing to centroid of home range
        self.distance_to_home_centroid = 0 # Bearing to centroid of home range
        self.target_location = (0,0) # Location of interest: place where fawn is resting, potential mate or something else.

        # Deer State
        self.is_fawn = False
        self.time_of_year = "Gestation" # ["Gestation","Fawning","PreRut","Rut"]
        self.behaviour_state = "Normal" # ["Normal","Disperse","Explore",""]
        self.speed_scaling_factor = 1 # Grid scaling factor. 1 == deer moves ~1 spatial unit per tick

        # Yaml Config Variables
        # Params is a global dict created in the model init: parameters.init_params(args.parameters_file, args.parameters)
        self.cfg = params.get('deer_control_vars')

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
            y_delta = abs(y_delta) # Make next Y component towards centroid
        elif last_y - centroid_y > home_range_d:
            y_delta = 0 - abs(y_delta) # Make next Y component towards centroid

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
    
    def check_time(self, tick_datetime):
        '''
        Update age from birthdate + tick
        Check what time of year it is:
            - Gestation ( 1 Jan - 14 May)
            - Fawning   (15 May - 31 Aug)
            - PreRut    ( 1 Sep - 31 Oct)
            - Rut       ( 1 Nov - 31 Dec)
        '''
        self.age = tick_datetime - self.birth_date
        if self.age > 100:
            self.is_fawn = False


        # time_of_year = 
        return
    
    def check_homerange(self, last_location):
        '''
        If in dispersal mode check if this can be a home range
        If in normal mode recalc direction to centroid
        '''
        
        # Movement/Location Vars
        self.has_home_range = True # Agent has a home range.
        self.home_range_centroid = (0,0) #Home range centroid x,y.  
        self.direction_to_home_centroid = None # Bearing to centroid of home range
        
        if self.has_home_range:
            pass
            # TODO:
            # self.direction_to_home_centroid = some function to get direction
            # self.distance_to_homes_centroid = some function for distance
        else:
            # TODO: 
            # No home range. Check if this could be a new home range. Probably a function of distance to other deer
            pass

        return
    
    def check_group(self):
        '''
        Check distance to herd group, distance to fawn, or distance to potential mate
        '''

        return

    def check_disease(self, tick):
        '''
        Check disease progression, whether this agent is infectious 
        '''
        return

    def calculate_next_state(self):
        '''
        Check whether the next state of this agent should be:
            - Normal: Do deer things
            - Disperse: Go in search of new home range
            - Mating: Follow a female during mating season
            - Explore: Go explore more. Less tight on home range
        '''
        return

    def step(self, last_location, tick_datetime):
        '''
        Function for the agent class to change behaviour state, and location. 

        Should be called in the model.step function. 
        ''' 
        
        self.check_time(tick_datetime)       # Check age of agent. Promote from Fawn to Adult, or go into Dispersal state 
        self.check_homerange(last_location)      # Check if this could be a home range, what the angle to the centroid is. 
        self.check_group()          # Check parent/fawn/group. 
        self.check_disease(tick_datetime)    # Check if infected and results thereof.

        self.calculate_next_state() # Figure out the next state for this agent

        next_x, next_y = self.calculate_next_pos(last_location.x, last_location.y, tick)
        return next_x, next_y