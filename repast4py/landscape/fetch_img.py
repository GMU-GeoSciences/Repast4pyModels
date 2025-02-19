from dataclasses import dataclass, field, fields, asdict
from owslib.wcs import WebCoverageService 
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
from rasterio.transform import array_bounds
import os
import shapely
# import pandas as pd
# import geopandas as gpd

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
            arr, transform = mask(ds, shapes = [bbox], crop = True)
            # Cropped Image Bounds
            image_bounds = array_bounds(arr.shape[-2], arr.shape[-1], transform)

            image_bounds = rasterio.coords.BoundingBox(
                                        image_bounds[0], 
                                        image_bounds[2], 
                                        image_bounds[1], 
                                        image_bounds[3])
            
            arr = reshape_as_image(arr)
             
    else:
        log.info('GeoTiff does not exist, going to download it...')

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


def csv_to_gpkg(params):
    '''
    Convert the repast csv logfile into a geopackage with 
    the x,y coords as point geom, and maybe the centroid geom too...
    '''
    in_csv = params['logging']['agent_log_file']
    out_gpkg = params['logging']['gpkg_log_file']

    # Load the CSV file
    df = pd.read_csv(in_csv)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df['x'], df['y'])
    )

    # Save to a GeoPackage file
    gdf.to_file(out_gpkg, layer='agent_location', driver='GPKG')
    return