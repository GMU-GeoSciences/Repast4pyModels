# Config Files

This config file is a work in progress. Much of it doesn't do anything and still needs to be implemented. Some of it is a duplicate of something else... Pinch of salt required.

Below is an example file with more comments added. 

The GEO section handles the geographic configuration of the model. The TIFF_PATH variable points to the raster to be loaded into the Repast shared data object.

```yaml
geo: 
  # Envelope for grid/continuous space for the simulation. Must be integers
  # Also the path to the geotiff to use.
  x_min: 1615000
  x_max: 1630000
  y_min: 1960000
  y_max: 1970000
  tiff_path: ./input/images/2019_CONUS_Canopy.tiff #in relation to project root dir
  buffer_size: 20 # max number of pixels that an agent can move in a single tick
```

The x/y boundary values are linked to the projection value of the raster file to be used. In the above example they refer meter values in the EPSG:5070 projection of the 2019 CONUS Canopy file. The *buffer_size* value is used by the Repast Shared Data Layer to create a buffer around the portions of the tiff given to different nodes. In practice this value should be higher than the maximum distance a deer could move in a single step. 

```yaml
time: 
  start_time: "2000-01-01T00:00:00" #ISO 8601 Timestamp
  hours_per_tick: 1
  end_tick: 300
```

This controls the start date (ISO 8601 Timestamps!) and the amount of time that passes for each step in the simulation. The last timstamp in a simulation is essentially *start_time* + *hours_per_tick* x *end_tick*.

```yaml
logging:
  loglevel: INFO # can be DEBUG, INFO, WARNING, ERROR
  agent_log_file: 'output/agent_log.csv'
```

This controls the verbosity of the logging in the simulation. For development work DEBUG is fine, but it should not be used when doing a full simulation. 

*agent_log_file* is the path to save the simulation outputs to, this is not the file containing the outputs from the python logger, but instead holds the output from the repast data logger.

```yaml
deer:
  pop_size: 300 # Number of deer agents to simulate
  disease_starting_ratio: 0 # What ratio of deer start sim being sick.
  input_agents: 'input/deer_agents.parq' # Starting params for each deer agent.
```

This is the number of deer agents to simulate, and the location of the file that holds agent initialisation info (not currently used).
Below is a config file used to test on a local machine:

```yaml
##########################
# Local Machine Simulation Config
# This config is applied using "make deer_run" on a non-hopper machine
########################## 
geo: 
  # Envelope for grid/continuous space for the simulation. Must be integers  
  # Small Corner of Howard County:
  x_min: 1615000
  x_max: 1630000
  y_min: 1962000
  y_max: 1974000

  tiff_path: ./input/images/2019_CONUS_Canopy.tiff #in relation to project root dir
  # tiff_path: ./input/images/RSF_suitability_classes_5070_resample.tif #in relation to project root dir
  buffer_size: 50  # Pixel overlap for the region covered by each process/core.
 
time: 
  start_time: "2000-01-01T00:00:00" #ISO 8601 Timestamp
  hours_per_tick: 1
  end_tick: 400 # End of sim = start_time + end_tick*hours_per_tick
 
logging:
  loglevel: INFO # Can be DEBUG, INFO, WARNING, ERROR

  ## Log Files:
  agent_log_file: './output/hmm3state_return_home.csv'
  model_meta_file: './output/hmm3state_return_home.json'
  gpkg_log_file: './output/hmm3state_return_home.gpkg'
  centroid_log_file: './output/hmm3state_return_home_centroid.gpkg'

# https://columbiaassociation.org/open-space/oh-deer-tips-and-tricks-for-living-with-deer-in-columbia/#:~:text=The%20Howard%20County%20Department%20of,15%20deer%20per%20square%20mile.
deer:
  pop_size: 10  # Number of deer agents to simulate  
  deer_vision_range: 400 # [meters]. Edge length of vision square surrounding agent. Used for finding nearby agents and home range suitability

deer_control_vars:  # From https://www.researchgate.net/publication/363077733_The_effect_of_landscape_transmission_mode_and_social_behavior_on_disease_transmission_Simulating_the_transmission_of_chronic_wasting_disease_in_white-tailed_deer_Odocoileus_virginianus_populations_usi
  movement_method: # Uncomment ONE of the below movement methods.
    ## Available methods are: DLD, HMM2, HMM3, HMM3_Canopy
    method: DLD # DLD method used to calculate behaviour states and movement distributions 
  explore_chance:
    gestation: 0.032
    fawning: 0.008
    prerut: 0.02
    rut: 0.096
    duration: 
      min: 12 
      max: 24 #Hours
  # gestation:  ## NOT IMPLEMENTED ## 
  #   min: 187 #Days
  #   max: 222 #Days
  #   fawn_prob:
  #     one: 0.25
  #     two: 0.5
  #     three: 0.25
  # grouping:  ## NOT IMPLEMENTED ## 
  #   adhesion:
  #     female: 
  #       normal: 0.95
  #       fawning: 0
  #     male:
  #       normal: 0.4
  #       rut: 0
  #   size_max:
  #     male: 10
  #     female: 4
  annual_mortality:
    female: 0.2
    male: 0.4
    fawn: 0.2
  age:
    fawn_to_adult: 360 # days. Fawn grows up
    adult_max: 100000 # days. Max age of agents
    male_disperse_prob: 0.7 # (Hawkins et al., 1971; Nixon et al., 2007, 1994; Rosenberry et al., 1999) 
    female_disperse_prob: 0.2 # (Hawkins et al., 1971; Nixon et al., 2007, 1994; Rosenberry et al., 1999) 
  homerange:
    radius: 2000 #[meters] Used in HMM_MoveModel_3_States_Canopy_Gender models. Agents beyond this range from home-range centroid choose angles that take them back home 
    turn_choices: 20 #Number of random choices to choose from. When agent is outside radius, create X number of turns and choose the one that is closest to return-to-centroid angle.
    max: 100
    min: 40
    suitability_threshold: 80
  disease:
  # Susceptible > Infected > Recovered/Dead > Susceptible
    infectious_range: 200 # NOTE: MUST be <= than 0.5*"deer_vision"! How close [meters] must an agent be to another infected agent to check for infection chance. 
    infectious_start_rate: 0.1 # portion of starting population that is infectious 
    infection_chance: 0.1 # Chance that agent will be infected upon single contact with infected agent. 
    infection_duration: (5,1) # normal(mean, stddev) Time after infection, until agent becomes recovered
    immunity_duration: (1000,1) # normal(mean, stddev) Time after recovery, until agent loses immunity 

```
