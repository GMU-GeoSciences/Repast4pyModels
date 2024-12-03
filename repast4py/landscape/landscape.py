import math 
from scipy.stats import exponweib, wrapcauchy
import numpy as np
from dataclasses import dataclass
import datetime

'''
This controls the landscape info. Reading a geotiff/WxS service
to turn the landscape data into a raster. 
'''
 
@dataclass
class Point:
    '''
    Hold X,Y and optionally Z coords.
    '''
    x: float
    y: float   
    z: float = 0.0