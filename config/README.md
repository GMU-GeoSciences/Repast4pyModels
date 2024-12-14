# Config Files

This config file is a work in progress. Much of it doesn't do anything and still needs to be implemented. Some of it is a duplicate of something else... Pinch of salt required.

Below is an example file with more comments added.

```yaml
sim:
 environment: local
```

The environment variable is deprecated this is handled in the MakeFile now...

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

The below values are taken from the DLD paper, but have not been implemented yet...

```yaml
deer_control_vars:  # From https://www.researchgate.net/publication/363077733_The_effect_of_landscape_transmission_mode_and_social_behavior_on_disease_transmission_Simulating_the_transmission_of_chronic_wasting_disease_in_white-tailed_deer_Odocoileus_virginianus_populations_usi
  male_disperse_prob: 0.7 # (Hawkins et al., 1971; Nixon et al., 2007, 1994; Rosenberry et al., 1999) 
  female_disperse_prob: 0.2 # (Hawkins et al., 1971; Nixon et al., 2007, 1994; Rosenberry et al., 1999) 
  explore_chance:
    gestation: 0.32
    fawning: 0.08
    prerut: 0.2
    rut: 0.96
    duration: 
      min: 12 
      max: 24 #Hours
  gestation:
    min: 187 #Daysx
    max: 222 #Days
    fawn_prob:
      one: 0.25
      two: 0.5
      three: 0.25
  grouping:
    adhesion:
      female: 
        normal: 0.95
        fawning: 0
      male:
        normal: 0.4
        rut: 0
    size_max:
      male: 10
      female: 4
  annual_mortality:
    female: 0.2
    male: 0.4
    fawn: 0.2
  annual_schedule: #ISO8601 --MM-DD .strftime('%m-%d')
    gestation_start: 01-01
    fawning_start: 05-15
    prerut_start: 09-01
    rut_start: 11-01
  age:
    fawn_to_adult: 12 #months
    adult_max: 100 #months. Max age of agents

```
