from dataclasses import dataclass, field, fields, asdict
from owslib.wcs import WebCoverageService 
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
import os
import shapely

import logging as pylog # Repast logger is called as "logging"

log = pylog.getLogger(__name__)

@dataclass
class WCS_Info:
    '''
    Simple class to hold info required to get image using Web Coverage Service 
    '''
    layer_id: str
    wcs_url: str 
    epsg: str
    description: str
    path: str
    bounds: [] #xmin, ymin, xmax, ymax 
    file_format: str = 'GeoTIFF'
    wcs_version: str = '2.0.1'
    x_label: str = 'X'
    y_label: str = 'Y'

def fetch_img(WCS_Info):
    '''
    Download WCS image, save it to a file, and return a numpy array of the raster.
    Also return the XY pixel resolution in the projection units.

    If the file already exists then just take a window snippet out of it for the area
    of interest.
    ''' 
    if os.path.isfile(WCS_Info.path):
        log.info('GeoTiff already exists, going to take snippet from it...') 
        with rasterio.open(WCS_Info.path, 'r') as ds:
            xy_resolution = ds.res
            image_bounds = ds.bounds
            bbox = shapely.geometry.box(WCS_Info.bounds[0], 
                                        WCS_Info.bounds[2], 
                                        WCS_Info.bounds[1], 
                                        WCS_Info.bounds[3])
            arr = reshape_as_image(mask(ds, shapes = [bbox], crop = True)[0]) 
            # Cropped Image Bounds
            image_bounds = rasterio.coords.BoundingBox(
                                        WCS_Info.bounds[0], 
                                        WCS_Info.bounds[2], 
                                        WCS_Info.bounds[1], 
                                        WCS_Info.bounds[3])
             
    else:
        log.info('GeoTiff does not exist, going to take download it...')

        wcs_service = WebCoverageService(WCS_Info.wcs_url,
                                version=WCS_Info.wcs_version)
        
        img = wcs_service.getCoverage(identifier = [WCS_Info.layer_id],
                                    srs = WCS_Info.epsg,
                                    subsets = [(WCS_Info.x_label, WCS_Info.bounds[0], WCS_Info.bounds[1]),
                                                (WCS_Info.y_label, WCS_Info.bounds[2], WCS_Info.bounds[3])],
                                    format = WCS_Info.file_format)
        
        out = open(WCS_Info.path, 'wb')
        out.write(img.read())
        out.close()

        with rasterio.open(WCS_Info.path, 'r') as ds:
            xy_resolution = ds.res
            image_bounds = ds.bounds
            arr = reshape_as_image(ds.read())   # read all raster values and return them as (x,y,z) 

    return arr, xy_resolution, image_bounds


# def bbox(coord_list):
#      box = []
#      for i in (0,1):
#          res = sorted(coord_list, key=lambda x:x[i])
#          box.append((res[0][i],res[-1][i]))
#      ret = f"({box[0][0]} {box[1][0]}, {box[0][1]} {box[1][1]})"
#      return ret