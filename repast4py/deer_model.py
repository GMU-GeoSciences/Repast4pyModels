import numpy as np
import pandas as pd
import repast4py
import logging as pylog # Repast logger is called as "logging"

from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass, field, fields, asdict

import collections
import csv

import random as rndm

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context 
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import BorderType, OccupancyType
from repast4py.space import DiscretePoint as dpt
from datetime import datetime, timedelta

# Local Files
from deer_agent.deer_agent import Deer, Deer_Config

# model = None
agent_tuple_cache = {}
log = pylog.getLogger(__name__)


def restore_agent(agent_tuple: Tuple):
    """
    Args:
        agent_tuple: tuple containing the data returned by save function.
        agent_tuple[0]: Agent ID
        agent_tuple[1]: Agent Type
        agent_tuple[2]: Rank
        agent_tuple[3+]: Internal state of agent

    Deer tuple example: ((5, 0, 1), CFG_Dict_Object)
    (5, 0 ,1) == (AgentID, Type, Rank)
    """
    log.debug(f'Restoring agent: {agent_tuple}')
    uid = agent_tuple[0] 
    if uid[1] == Deer.TYPE: #This should always be true
        if uid in agent_tuple_cache:
            # Agent exists in cache.
            agent = agent_tuple_cache[uid]
        else:
            # Cache this agent
            agent = Deer(uid[0], uid[2]) #(agent_id, rank)
            agent_cfg = agent_tuple[1]
            agent.set_cfg(agent_cfg)
            agent_tuple_cache[uid] = agent

        agent.cfg = agent_tuple[1]
    else:
        log.warning('Unknown agent type.')
        agent = None
    return agent

@dataclass
class Metrics:
    '''
    Dataclass used by repast4py aggregate logging to record
    some metrics after each tick.

    Currently recording:
        - agent id: something to identify the agent. Might be a combination of rank and ID
        - tick timestamp: TODO: Convert the tick to a timestamp
        - Behaviour state: What was the agent doing? Foraging/resting/etc?
        - Location: X/Y location of agent at tick time.

    '''
    agent_id: int = 0
    tick_timestamp: str = datetime.fromtimestamp(0).isoformat()
    agent_location_x: float = 0
    agent_location_y: float = 0
    agent_behaviour_state: int = 0

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
        self.contexts["deer"] = context.SharedContext(comm) # https://repast.github.io/repast4py.site/guide/user_guide.html#_contexts_and_projections
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)  

        self.runner.schedule_stop(params['time']['end_tick'])
        self.runner.schedule_end_event(self.end_sim)

        self.start_timestamp = datetime.fromisoformat(params['time']['start_time'])
        self.tick_time = self.start_timestamp
        self.tick_hour_interval = params['time']['hours_per_tick']
         
        # Setup the spatial side of the sim.
        x_height = int(params['geo']['x_max']) - int(params['geo']['x_min'])
        y_height = int(params['geo']['y_max']) - int(params['geo']['y_min'])

        bbox = space.BoundingBox(int(params['geo']['x_min']), 
                                x_height, 
                                int(params['geo']['y_min']), 
                                y_height, 
                                0, 
                                0)
        
        self.space = space.SharedCSpace('space', 
                                        bounds=bbox, 
                                        borders=BorderType.Sticky, 
                                        occupancy=OccupancyType.Multiple,
                                        buffer_size=2, 
                                        comm=comm, 
                                        tree_threshold=100)
        
        self.contexts["deer"].add_projection(self.space)

        # Setup the instrumentation to measure the outputs of the sim
        self.metrics = Metrics()
        self.agent_logger = logging.TabularLogger(comm, 
                                                  self.params.get('logging',{}).get('agent_log_file'), 
                                                  ['tick', 'agent_id', 'agent_uid_rank', 'timestamp', 'x', 'y', 'state'])

        
        # Initialise agents
        total_agent_count = self.params.get('deer',{}).get('pop_size')
        world_size = comm.Get_size()
        agents_per_rank = int(total_agent_count / world_size)
        if self.rank < total_agent_count % world_size:
            agents_per_rank += 1

        log.debug(f"Creating {total_agent_count} deer agents over {world_size} nodes ({agents_per_rank} agents/node)...")
        
        ## Creating count_per_rank agents on this node
        for this_agent_id in range(agents_per_rank):
            self.add_agent(this_agent_id) 

        self.last_agent_id = this_agent_id

    def start_sim(self):
        '''
        Start the agents and model.  
        '''
        log.debug('Starting model...')
        self.runner.execute()

    def end_sim(self):
        '''
        Clean up and log data.
        '''
        log.debug(f'Cleaning up simulation...')
        # self.data_set.close()
        #TODO: figure out logging. Is it going to be [time,x,y,data] ? 

    def add_agent(self, id, x = None, y = None):
        '''
        Add agent to the sim located at Point(x,y).
        If no location specified then place randomly in local bounds.
        '''
        local_bounds = self.space.get_local_bounds()    
        if (x is None) or (y is None): 
            x = random.default_rng.uniform(local_bounds.xmin, local_bounds.xmin + local_bounds.xextent)
            y = random.default_rng.uniform(local_bounds.ymin, local_bounds.ymin + local_bounds.yextent)
        
        agent = Deer(id, self.rank)
        #################
        # TODO: Init the models: 
        deer_cfg = Deer_Config()  
        deer_cfg_object = deer_cfg.rand_factory(params)
        agent.set_cfg(deer_cfg_object)
        #################
        log.debug(f'Creating Rank:Agent {self.rank}:{agent.id}.') 
        self.contexts['deer'].add(agent)
        self.move_agent(agent,x,y)

    def remove_agent(self, agent):
        '''
        Remove an agent from the sim. 
        '''
        log.debug(f'Removing agent...')
        self.context.remove(agent)
        return

    def move_agent(self, agent, x, y):
        '''
        Agents move in space over one tick
        '''
        log.debug(f'Moving agent {self.rank}:{agent.id} to Point({x},{y})...')
        self.space.move(agent, cpt(x,y))
        return

    def step(self):
        '''
        Increment the sim by one step.
        '''
        log.debug(f'Stepping...')
        tick = self.runner.schedule.tick
        self.tick_time = self.start_timestamp + timedelta(hours=self.tick_hour_interval)*tick
        self.log_metrics(tick) 
        self.contexts['deer'].synchronize(restore_agent)

        log.debug('Looping through agents')
        for agent in self.contexts['deer'].agents(Deer.TYPE):
            log.debug(f'Working with agent {self.rank}:{agent.id}')
            #For each DEER agent do something:
            location = model.space.get_location(agent)
            next_x,next_y = agent.step(location, self.tick_time)
            self.move_agent(agent, next_x, next_y)
    
    def log_metrics(self, tick):
        '''
        Log the outputs of the tick to a file.
        '''
        log.debug(f'Logging metrics...') 
        num_agents = self.contexts['deer'].size([Deer.TYPE])

        if tick % 10 == 0:
            log.info(f"  - Tick: {tick}: Agent count: {num_agents}")
            log.info(f"  - Timestamp: {self.tick_time.isoformat()}")

        for agent in self.contexts['deer'].agents():
            location = model.space.get_location(agent)
            # Scale location back to a normal projection. 
            # Table: ['tick', 'agent_id', 'agent_uid_rank', 'x', 'y', 'state']
            self.agent_logger.log_row(tick, 
                                      agent.id, 
                                      agent.uid_rank,
                                      self.tick_time.isoformat(),
                                      location.x,
                                      location.y,
                                      agent.behaviour_state)

        self.agent_logger.write()
        return
 

def setup_logging(params):
    '''
    Setup the logging for both Repast4Py as well as a normal python logger
    '''

    # Setup a command line logger.
    loglevel = params['logging']['loglevel']
    pylog.basicConfig(level=loglevel, format='%(asctime)s %(relativeCreated)6d %(threadName)s ::%(levelname)-8s %(message)s')
    log = pylog.getLogger()

    if params.get('sim',{}).get('environment') == 'hopper':
        # Only log ERROR's
        log.setLevel('ERROR')

    elif params.get('sim',{}).get('environment') == 'local':
        # Set Log level
        log.setLevel(loglevel)
    else:
        # Do not setup a command line logger.
        log.setLevel('ERROR')
        log.error('Unknown Environment...')

    log.debug('Ready to log!')
    log.info(f'Params: {params}')
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
