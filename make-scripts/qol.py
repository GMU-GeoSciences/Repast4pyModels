import pandas as pd
import geopandas as gpd
import yaml

'''
Convert the repast csv logfile into a geopackage with 
the x,y coords as point geom, and maybe the centroid geom too...
'''
in_csv = '../output/agent_log.csv'
agent_gpkg = '../output/agents.gpkg'
centroid_gpkg = '../output/centroids.gpkg'

# Load the CSV file
df = pd.read_csv(in_csv)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['x'], df['y'], crs='EPSG:5070')
)

# Save to a GeoPackage file
gdf.to_file(agent_gpkg, layer='agent_location', driver='GPKG')


# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df['centroid_x'], df['centroid_y'], crs='EPSG:5070')
)

# Save to a GeoPackage file
gdf.to_file(centroid_gpkg, layer='centroid_location', driver='GPKG')
    