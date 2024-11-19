import math 
from scipy.stats import exponweib, wrapcauchy
import numpy as np
from dataclasses import dataclass
import datetime

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
    
    def calc_point(self, intial_point, step_distance, turn_angle):
        '''
        When given a distance and angle calculate the X and Y coords of it
        when starting from a current position. 

        Turn Angle = Angle(Prev, Current) - Angle(Current, Next)
        '''
        next_x = intial_point.x + step_distance*np.sin(np.deg2rad(turn_angle))
        next_y = intial_point.y + step_distance*np.cos(np.deg2rad(turn_angle))
        
        next_point = Point(next_x,next_y)
        return next_point

    def step(self, step_distance, turn_angle):
        next_point =  self.calc_point(self.current_point, step_distance, turn_angle)
        # Making step
        self.last_point = self.current_point
        self.current_point = next_point

        self.calc_dist_and_angle()
        
        return next_point

class Movement:
    """Class for creating next step from current step and timestamp. """  
    
    def __init__(self, pos_vector : Position_Vector, timestamp: datetime.datetime):
        # Calculate season from timestamp
        # Look up weibull and cauchy params for season
        # Calculate step and angle from PDF
        # Convert step and angle to (x,y) from position_vector
        # Return next (x,y)

        self.pos = pos_vector
        self.timestamp = timestamp 

        # Default weibull_params from all GPS points
        self.a = 6.116
        self.c = 0.385
        self.loc = 0.039 
        self.scale = 5.640

    def step(self):
        '''
        When given a distance and angle calculate the X and Y coords of it
        when starting from a current position. 
        '''
        step_distance = self.calculate_random_step()
        step_angle = self.calculate_random_turn()
        self.pos.step(step_distance, step_angle)
        next_position = self.pos.current_point
        return next_position
 
    def calculate_random_step(self):
        '''
        Takes timestamp, extracts month/hour from it and chooses correct
        mode of stepping.

        TODO: This needs to be generalised/not hard coded... 
        '''
        this_season = get_season(self.timestamp)

        if this_season == 'Gestation':
            # These are calculated from the GPS data similar to how the DLD paper does it.
            a = 7.441
            c = 0.358
            loc = -0.008
            scale = 3.717
        elif this_season == 'Fawning':
            a = 5.265
            c = 0.425
            loc = -0.017  
            scale = 7.519          
        elif this_season == 'PreRut':
            a = 4.478
            c = 0.429
            loc = -0.00719
            scale = 10.923116
        elif this_season == 'Rut': 
            a = 5.354084
            c = 0.383384
            loc = 0.101771
            scale = 7.730135
        else: 
            a = self.a
            c = self.c
            loc = self.loc
            scale = self.scale

        step_distance = exponweib.rvs(a, c, loc, scale)

        return step_distance
    
    def calculate_random_turn(self):
        '''
        This calculates a random step by creating a random distribution
        using the distance and turn angle to the centroid. 

        This is an implementation of the "simple return" distribution from DLD paper. 
        TODO: Needs to handle the case where the agent has no home range/ no home range centroid.

        for stats package:
            c = rho_t
            x = u_t 
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wrapcauchy.html#scipy.stats.wrapcauchy
        '''
        this_season = get_season(self.timestamp)
        distance_from_centroid = self.pos.distance_to_centroid

        u_t = np.deg2rad(self.pos.heading_to_centroid)
        p_t = 0.5 

        turn_angle = wrapcauchy.rvs(p_t, loc = u_t, scale = 1) 
        return np.rad2deg(turn_angle)


def get_season(date):
    """
    Groups a date into a season based on given start and end dates.
    Args:
        date: The date to classify. 
    Returns:
        The name of the season.
    """
    season_names = ['Gestation','Fawning','PreRut','Rut']
    start_dates = [(1, 1), (5, 15), (9, 1), (11, 1)]
    end_dates = [(5, 14), (8, 31), (10, 31), (12, 31)]

    month, day = date.month, date.day

    for i in range(len(start_dates)):
        start_month, start_day = start_dates[i]
        end_month, end_day = end_dates[i]

        if start_month < end_month:
            if start_month <= month <= end_month:
                if (month == start_month and day >= start_day) or (month == end_month and day <= end_day) or (start_month < month < end_month):
                    return season_names[i]
        else:  # Handle seasons that span across year-end
            if (month >= start_month and day >= start_day) or (month <= end_month and day <= end_day):
                return season_names[i]
    return "Unknown Season"