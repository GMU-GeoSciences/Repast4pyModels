# Repast4pyModels

## Deer Model

Currently the deer simulation is very simple:

- Deer agents are created with random features using deer_agent.Deer_Config()
- The agent is moved to a random location
- The first location of the agent is considered the home range centroid
- The agent moves randomly around by calculating step distance and turn angles from deer_agent.movement.Movement class
- The agent logs it's location and the value for the loaded raster at that point

That's it. Nothing too fancy but a good place to iterate from...
