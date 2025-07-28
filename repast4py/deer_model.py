import logging as pylog # Repast logger is called as "logging"

from typing import Dict, Tuple
import deer_agent.agent
from mpi4py import MPI

from copy import copy, deepcopy
import json
import os
import pathlib

import random as rndm
import math
import torch

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context 
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import BorderType, OccupancyType
from repast4py.space import DiscretePoint as dpt
from repast4py.value_layer import SharedValueLayer
from datetime import datetime, timedelta

# Local Files 
import deer_agent
from landscape import fetch_img, landscape_funcs 

from deer_agent.movement_basic import BaseMoveModel, RandomMovement
from deer_agent.movement_dld import DLD_MoveModel
from deer_agent.movement_hmm import HMM_MoveModel_2_States, HMM_MoveModel_3_States, HMM_MoveModel_3_States_Canopy_Gender
from deer_agent import movement
from deer_agent.disease import *
from deer_agent.time_functions import *


# model = None
agent_tuple_cache = {}
log = pylog.getLogger(__name__)

'''
Some Examples:
https://github.com/Meguazy/Multi-agent-systems-and-parkinson/blob/main/src/parkinson.py 
'''

def restore_agent(agent_tuple: Tuple):
    """
    Deer tuple example: ((5, 0, 1), CFG_Dict_Object)
    (5, 0 ,1) == (AgentID, Type, Rank)

    Args:
        agent_tuple: tuple containing the data returned by save function.
        agent_tuple[0][0]: Agent ID
        agent_tuple[0][1]: Agent Type
        agent_tuple[0][2]: Rank
        agent_tuple[1]: CFG Dict 
    """
    log.debug(f'Restoring agent: {agent_tuple}')
    uid = agent_tuple[0]
    # agent = Deer(uid[0], uid[2]) #(agent_id, rank)
    # agent_cfg = agent_tuple[1]
    # agent.set_cfg(agent_cfg)

    if uid[1] == deer_agent.agent.DeerAgent.TYPE: 
        if uid in agent_tuple_cache:
            # Agent exists in cache.
            agent = agent_tuple_cache[uid]
        else:
            # Cache this agent
            agent = deer_agent.agent.DeerAgent(uid[0], uid[2]) #(agent_id, rank)
            agent_cfg = agent_tuple[1]
            agent.set_cfg(agent_cfg)
            agent_tuple_cache[uid] = agent
 
        agent_cfg = agent_tuple[1]
        agent.set_cfg(agent_cfg)
    else:
        log.warning('Unknown agent type.')
        agent = None

    return agent

# @dataclass
# class Metrics:
#     '''
#     Dataclass used by repast4py aggregate logging to record
#     some metrics after each tick.

#     Currently recording:
#         - agent id: something to identify the agent. Might be a combination of rank and ID
#         - tick timestamp: TODO: Convert the tick to a timestamp
#         - Behaviour state: What was the agent doing? Foraging/resting/etc?
#         - Location: X/Y location of agent at tick time.

#     '''
#     tick_timestamp: str = datetime.fromtimestamp(0).isoformat()
#     agent_id: int = 0
#     agent_location_x: float = 0
#     agent_location_y: float = 0
#     agent_centroid_x: float = 0
#     agent_centroid_y: float = 0
#     suitable: str = '' 
#     agent_behaviour_state: str = ''

class Model:
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behaviour of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """
    contexts = {}
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        '''
        Initialise the Model Class
        '''
        log.debug('Initialising Repast model...')
        self.params = params
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.contexts['deer'] = context.SharedContext(comm) # https://repast.github.io/repast4py.site/guide/user_guide.html#_contexts_and_projections
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)  

        self.runner.schedule_stop(params['time']['end_tick'])
        self.runner.schedule_end_event(self.end_sim)

        self.start_timestamp = datetime.datetime.fromisoformat(params['time']['start_time'])
        self.tick_time = self.start_timestamp
        self.tick_hour_interval = params['time']['hours_per_tick']
        
        # Setup the grids, read a raster file in, 
        # and transfer it into a shared_value array
        self.setup_repast_spatial()
        self.setup_logging()
 
        # Initialise agents
        ##################################################
        total_agent_count = self.params.get('deer',{}).get('pop_size')
        world_size = comm.Get_size()
        agents_per_rank = int(total_agent_count / world_size)
        if self.rank < total_agent_count % world_size:
            agents_per_rank += 1
        log.debug(f"Creating {total_agent_count} deer agents over {world_size} nodes ({agents_per_rank} agents/node)...")
        
        ## Creating count_per_rank agents on this node
        this_agent_id = 0
        while this_agent_id < agents_per_rank:
            #keep trying to add agents into the simulation until the correct amount is achieved. 
            agent = self.add_agent(this_agent_id)
            deer_vision_range = int(self.params['deer']['deer_vision_range'])  
            grid_range = math.ceil(200 / self.xy_resolution[0])
            x_es = range(int(agent.centroid_x) - grid_range, int(agent.centroid_x) + grid_range)
            y_es = range(int(agent.centroid_y) - grid_range, int(agent.centroid_y) + grid_range)
            local_array = np.zeros(shape=(len(x_es), len(y_es)), dtype=int) # Empty array for sense_range pixels around the agent
            for i in x_es:
              for j in y_es:  
                local_array[i - min(x_es), j - min(y_es)] = self.canopy_layer.get(dpt(i,j)).item()
                
            good_hr = location_suitability(local_array, params) 
            if good_hr:
                self.contexts['deer'].add(agent)
                self.move_agent(agent,agent.centroid_x,agent.centroid_y)
                this_agent_id += 1
            else:
                agent = None
        ##################################################

 

        
    def setup_repast_spatial(self):
        '''
        Return a shared grid, shared continuous space and a shared value layer/s
        that contain a space for agents to move on and
        a grid for agents to look for neighbours on (similar to zombie example)
        and a shared layer to hold geotiff data. 

        Because of limitation on the way repast handles grids the grid will be
        scaled to the raster resolution. 
        ''' 
        log.debug('Initialising Repast spatial grids...')
        canopy = fetch_img.WCS_Info(layer_id='mrlc_download:nlcd_tcc_conus_2019_v2021-4',
                 wcs_url = 'https://www.mrlc.gov/geoserver/ows', 
                 epsg = 'EPSG:5070',
                 path = params['geo']['tiff_path'],
                 bounds = [int(params['geo']['x_min']), 
                           int(params['geo']['x_max']), 
                           int(params['geo']['y_min']), 
                           int(params['geo']['y_max'])], 
                 description = '2019 NLCD Canopy Estimate for Howard County'
                 )

        # xy_resolution and image_bounds are going to get overloaded if multiple rasters are ingested from different sources
        # Either stick to a single source, a single raster, or figure this out...
        canopy_array, self.xy_resolution, self.image_bounds = fetch_img.fetch_img(canopy)

        log.info(f'WCS Array shape: {canopy_array.shape}')
        log.info(f'WCS Image Bounds: {self.image_bounds}')

        # Convert projection units to pixel units:
        projection_bounds = space.BoundingBox(0, 
                                    canopy_array.shape[1], 
                                    0, 
                                    canopy_array.shape[0], 
                                    0, 
                                    0) # Canopy_array returned as (y,x,z) >> (rows, columns, bands) of geotiff
        log.info(f'Total Projection bounds: {projection_bounds}')
        # https://repast.github.io/repast4py.site/apidoc/source/repast4py.value_layer.html#repast4py.value_layer.SharedValueLayer
        self.canopy_layer = SharedValueLayer(comm = self.comm, 
                                        bounds = projection_bounds, 
                                        borders = space.BorderType.Sticky, 
                                        buffer_size = int(params['geo']['buffer_size']), 
                                        init_value = 0)

        log.info(f'This Rank Projection bounds: {self.canopy_layer.bounds}')

        xy = [self.canopy_layer.bounds.ymin, 
              self.canopy_layer.bounds.ymin + self.canopy_layer.bounds.yextent,
              self.canopy_layer.bounds.xmin, 
              self.canopy_layer.bounds.xmin + self.canopy_layer.bounds.xextent ] 
         
        sub_array = canopy_array[xy[0]:xy[1],xy[2]:xy[3],0] 
        self.canopy_layer.grid[:,:] = torch.from_numpy(sub_array).type(torch.float64)

        self.grid = space.SharedGrid('grid', 
                                bounds = projection_bounds, 
                                borders = space.BorderType.Sticky, 
                                occupancy = OccupancyType.Multiple,
                                buffer_size = int(params['geo']['buffer_size']), 
                                comm=self.comm)

        self.shared_space = space.SharedCSpace('space', 
                                bounds = projection_bounds, 
                                borders = BorderType.Sticky, 
                                occupancy = OccupancyType.Multiple,
                                buffer_size = int(params['geo']['buffer_size']), 
                                comm=self.comm, 
                                tree_threshold=100)

        self.contexts['deer'].add_projection(self.grid) 
        self.contexts['deer'].add_projection(self.shared_space)
        self.contexts['deer'].add_value_layer(self.canopy_layer)

    def start_sim(self):
        '''
        Start the agents and model.  
        '''
        log.info('Starting model...')
        movement_method = self.params['deer_control_vars']['movement_method']['method']
        log.info(f'Using movement model:{movement_method}') 
        self.server_start_timestamp = datetime.datetime.now()
        self.runner.execute()

    def end_sim(self):
        '''
        Clean up and log data.
        '''
        log.info(f'Cleaning up simulation...')
        self.end_start_timestamp = datetime.datetime.now()
        self.agent_logger.close()
        self.log_model_info() 
        return
    

    def log_model_info(self):
        '''
        Log the model info like the area covered, number of agents, time it took to run etc
        The goal is to have performance metadata that would be used in benchmarking.
        '''
        log.info("Saving model run info...")
        x_km = self.xy_resolution[0]*(self.image_bounds.right - self.image_bounds.left)/1000
        y_km = self.xy_resolution[1]*(self.image_bounds.top - self.image_bounds.bottom)/1000

        output_dict = {
                    'number_of_ranks':self.comm.Get_size(),
                    'agents_at_start': self.params.get('deer',{}).get('pop_size'),
                    'agent_at_end':'',
                    'spatial_km2':x_km*y_km,
                    'model_start_time':self.start_timestamp.isoformat(),
                    'model_end_tick':self.params['time']['end_tick'],
                    'hours_per_tick':self.tick_hour_interval,
                    'server_start_time':self.server_start_timestamp.isoformat(),
                    'sever_end_time':self.end_start_timestamp.isoformat(),
                    'sim_duration_minutes': (self.end_start_timestamp - self.server_start_timestamp).total_seconds() / 60.0,
                    'agents_per_km2':int(self.params.get('deer',{}).get('pop_size'))/(x_km*y_km),
                    'params': self.params,
                    }
        
        log_path = pathlib.Path(self.params['logging']['model_meta_file']).parents[0]
        log_path.mkdir(parents=True, exist_ok=True, mode=0o666)
        with open(self.params['logging']['model_meta_file'], 'w') as f:
            json.dump(output_dict, f)
        os.chmod(self.params['logging']['model_meta_file'], 0o666)
        
        return output_dict 

    def add_agent(self, id, x = None, y = None):
        '''
        Add agent to the sim located at Point(x,y).
        If no location specified then place randomly in local bounds.
        '''
        local_bounds = self.shared_space.get_local_bounds()
        if (x is None) or (y is None): 
            # If no coords given, place it semi-randomly. 
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
            
        agent = deer_agent.agent.DeerAgent(id, self.rank) 
        deer_cfg = deer_agent.agent.Deer_Config()  
        deer_cfg_object = deer_cfg.rand_factory(x,y,params)
        agent.set_cfg(deer_cfg_object)

        ## Assign first place as agent home range centroid. Need to find a better way of doing this.
        agent.centroid_x = x
        agent.centroid_y = y
        log.debug(f'Creating Rank:Agent {self.rank}:{agent.id}.') 
        return agent

    def remove_agent(self, agent):
        '''
        Remove an agent from the sim. 
        '''
        log.debug(f'Removing dead agent: {agent.uuid}')
        self.contexts['deer'].remove(agent)
        return

    def move_agent(self, agent, x, y):
        '''
        Agents move in space over one tick
        '''
        log.debug(f'Moving agent {self.rank}:{agent.id} to Point({x},{y})...')
 
        self.shared_space.move(agent, cpt(x,y))
        self.grid.move(agent, dpt(int(x),int(y)))

        location = self.shared_space.get_location(agent)
        agent.current_x = copy(location.x)
        agent.current_y = copy(location.y) 
        return

    def step(self):
        '''
        Increment the sim by one step.
        '''
        tick = self.runner.schedule.tick
        log.debug(f' ==== Stepping for tick: {tick} ==== ')
        self.tick_time = self.start_timestamp + timedelta(hours=self.tick_hour_interval)*tick
        self.contexts['deer'].synchronize(restore_agent)
        log.debug('Looping through agents')
        dead_agents = []
        deer_vision_range = int(self.params['deer']['deer_vision_range'])

        for agent in self.contexts['deer'].agents(deer_agent.agent.DeerAgent.TYPE):
            log.debug(f'Working with agent {self.rank}:{agent.id}')

            # Get tick info
            agent.timestamp = self.tick_time
            # local_cover, other_agents = landscape.get_nearby_items(agent, model, sense_range=deer_vision_range)
            
            # Calculate disease state
            # nearby_agents = landscape.get_nearby_agents(agent, model, sense_range = deer_vision_range)
            # This nearby_agnets call is causing trouble with the HMM models... Why?  
            nearby_agents = landscape_funcs.get_nearby_grid_agents(agent, self, sense_range = deer_vision_range)
            agent = check_disease_state(agent, nearby_agents, params, resolution = self.xy_resolution[0])

            # Calculate behaviour states, and movement based off a 
            # model method specified in the config.
            # TODO: these movement models should have a single style of input/output
            # and be attached to the agent class during agent init.
            movement_method = self.params['deer_control_vars']['movement_method']['method']
            if movement_method == 'basic':
                '''
                agents do not move.
                '''
                movement_model = BaseMoveModel()
                agent.step_distance, agent.turn_angle = movement_model.step()
                agent.calc_next_point(self.xy_resolution, self.image_bounds)#agent.step_distance, agent.turn_angle) 

            elif movement_method == 'random':
                '''
                Agents move randomly without a care for landscape
                behaviour states or time of day/year.
                '''
                movement_model = RandomMovement()
                agent.step_distance, agent.turn_angle = movement_model.step()
                agent.calc_next_point(self.xy_resolution, self.image_bounds)#agent.step_distance, agent.turn_angle) 

            elif movement_method == 'HMM2':
                '''
                Use a hidden markov model to calculate behaviour states
                and movement parameters.
                '''
                move_model = HMM_MoveModel_2_States() 

                local_cover = landscape_funcs.get_nearby_pixels(agent, model, sense_range = deer_vision_range)
                # local_cover = np.ones((deer_vision_range,deer_vision_range))

                next_state, step_distance, turn_angle  = move_model.step(agent, local_cover)
                agent.step_distance = step_distance
                agent.turn_angle = turn_angle

                agent.calc_next_point(self.xy_resolution, self.image_bounds)#step_distance, turn_angle) 
                agent.behaviour_state = next_state 


            elif movement_method == 'HMM3':
                '''
                Use a hidden markov model to calculate behaviour states
                and movement parameters.
                '''
                move_model = HMM_MoveModel_3_States() 

                local_cover = landscape_funcs.get_nearby_pixels(agent, model, sense_range = deer_vision_range)

                next_state, step_distance, turn_angle  = move_model.step(agent, local_cover)
                agent.step_distance = step_distance
                agent.turn_angle = turn_angle

                agent.calc_next_point(self.xy_resolution, self.image_bounds)#step_distance, turn_angle) 
                agent.behaviour_state = next_state 

            elif movement_method == 'HMM3_Canopy':
                '''
                Use a hidden markov model to calculate behaviour states
                and movement parameters. That considers the CAnopy raster as the covariate.
                '''
                move_model = HMM_MoveModel_3_States_Canopy_Gender()
                move_model.home_range_radius = int(self.params['deer_control_vars']['homerange']['radius']) 
                move_model.turn_choices =  int(self.params['deer_control_vars']['homerange']['turn_choices']) 

                local_cover = landscape_funcs.get_nearby_pixels(agent, model, sense_range = deer_vision_range)

                next_state, step_distance, turn_angle  = move_model.step(agent, local_cover)
                agent.step_distance = step_distance
                agent.turn_angle = turn_angle
                
                agent.calc_next_point(self.xy_resolution, self.image_bounds)
                agent.behaviour_state = next_state 

            elif movement_method == 'DLD': 
                # Calculate next step

                local_cover = landscape_funcs.get_nearby_pixels(agent, model, sense_range = deer_vision_range)
 
                # Get tick info
                agent.timestamp = self.tick_time 
                
                # Calculate disease state
                
                #Use DLD Methods of states and steps
                # Calculate next step
                agent = behaviour.calculate_next_state(agent, local_cover, nearby_agents, params)
                next_position = movement.step(agent, self.xy_resolution, self.image_bounds)  

            else:
                log.error('Unknown movement choice')
                break 
            self.move_agent(agent, agent.current_x, agent.current_y)
            if agent.is_dead:
                dead_agents.append(agent)

        for agent in dead_agents:
            self.remove_agent(agent)
        
        self.log_metrics(tick)
        return
    
    def setup_logging(self):
        # Setup the instrumentation to measure the outputs of the sim 
        self.agent_logger = logging.TabularLogger(self.comm, 
                                                  self.params.get('logging',{}).get('agent_log_file'), 
                                                  ['Timestamp', 
                                                   'UUID',  
                                                #    'UID',
                                                #    'Rank',
                                                   'Is Male',
                                                #    'Is Fawn',
                                                #    'Has HomeRange',
                                                #    'x_array', 'y_array', 
                                                   'x', 'y', 
                                                #    'x_c_array', 'y_c_array', 
                                                   'x_centroid', 'y_centroid', 
                                                   'step', 'turn',  
                                                   'heading_from_prev', 
                                                   'turn_to_centroid',
                                                   'heading_to_centroid',
                                                   'Behaviour State', 
                                                   'Disease State',
                                                #    'grid_location',
                                                #    'NearbyOtherAgentCount',
                                                #    'step_distance',
                                                #    'turn_angle'
                                                   ])
        return
    

    def log_metrics(self, tick):
        '''
        Log the outputs of the tick to a file.
        '''
        log.debug(f'Logging metrics...') 
        # num_agents = self.contexts['deer'].size([Deer.TYPE])

        if tick % 24 == 0: #Log once per day/24 ticks
            # log.info(f"  - Tick: {tick}: Agent count: {num_agents}")
            log.info(f"  - Timestamp: {self.tick_time.isoformat()}")

        for agent in self.contexts['deer'].agents():
            cont_location = self.shared_space.get_location(agent)
            # grid_location = self.grid.get_location(agent) 

            if cont_location is not None:
                x = cont_location.x  # There is a small offset between raster values and agent locations shown in QGIS
                y = cont_location.y  
            else:
                x = -1
                y = -1

            # Scale location back to a 5070 projection.   
            x_proj, y_proj = agent.repast_to_proj(x,y, self.xy_resolution, self.image_bounds)
            # Get centroid and project it to 5070
            x_proj_centroid, y_proj_centroid = agent.repast_to_proj(agent.centroid_x, agent.centroid_y,self.xy_resolution, self.image_bounds) 
            turn_to_centroid = agent.heading_from_prev - agent.heading_to_centroid #Calculate the turn angle that would take you back to the centroid

            log.debug(' -- Writing to log...')
            self.agent_logger.log_row(self.tick_time.isoformat(),
                                      agent.uuid, 
                                    #   agent.uid, 
                                    #   agent.uid[2], #rank
                                      agent.is_male,
                                    #   x,
                                    #   y,
                                      x_proj,
                                      y_proj,
                                    # agent.centroid_x,
                                    # agent.centroid_y,
                                      x_proj_centroid,
                                      y_proj_centroid,
                                     agent.step_distance*self.xy_resolution[0],
                                     np.degrees(agent.turn_angle),
                                     np.degrees(agent.heading_from_prev), 
                                     np.degrees(turn_to_centroid),
                                     np.degrees(agent.heading_to_centroid),
                                    #   suitable, 
                                      agent.behaviour_state, 
                                      agent.disease_state,
                                      #Temp data
                                    #   str(grid_location),
                                    #   len(nearby_agents),
                                    #   agent.step_distance*self.xy_resolution[0],
                                    #   agent.pos.turn_angle
                                      )
    
        self.agent_logger.write()
        log.debug(' -- Finished writing to log.')
        return

def setup_logging(params):
    '''
    Setup the logging for both Repast4Py as well as a normal python logger
    ''' 
    # Setup a command line logger.
    loglevel = params['logging']['loglevel']
    pylog.basicConfig(level=loglevel, format='%(asctime)s %(relativeCreated)6d %(threadName)s ::%(levelname)-8s %(message)s')
    log = pylog.getLogger() 

    log.debug('Ready to log!')
    log.debug(f'Params: {params}')
    return log

def run(params: Dict):
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.start_sim()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    setup_logging(params)
    run(params)
    log.info('==END==')
