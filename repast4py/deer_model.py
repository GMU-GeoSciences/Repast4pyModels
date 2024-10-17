import numpy as np
import pandas as pd
import repast4py
import logging as pylog # Repast logger is called as "logging"

from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass
import collections
import csv

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context 
from repast4py.space import ContinuousPoint as cpt
from repast4py.space import BorderType, OccupancyType
from repast4py.space import DiscretePoint as dpt
from datetime import datetime, timedelta

# Local Files
from agents import Deer

model = None
agent_tuple_cache = {}
def restore_agent(agent_tuple: Tuple):
    """
    Args:
        agent_tuple: tuple containing the data returned by save function.
        agent_tuple[0]: Agent ID
        agent_tuple[1]: Agent Type
        agent_tuple[2]: Rank
        agent_tuple[3+]: Internal state of agent

    TODO: Test Caching. Maybe try pyfuncs decorator? 
    Does the caching make sense? I can't really see how it would make any difference to update vs recreate a tuple. 
    TODO: Make flexible for multiple agent types
    """
    uid = agent_tuple[0]  

    if uid[1] == Deer.TYPE:
        if uid in agent_tuple:
            # Agent exists in cache.
            agent = agent_tuple_cache[uid]
        else:
            # Create agent from tuple
            agent = Deer(uid[0], uid[2], pt) # Does this
            agent_tuple_cache[uid] = agent

        agent.pt = pt
    else:
        pylog.warning('Unknown agent type.')
        agent = None
    return agent

@dataclass
class Metrics:
    '''
    Dataclass used by repast4py aggregate logging to record
    some metrics after each tick.
    '''
    deer: int = 0

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
        pylog.debug('Initialising Repast model...')

        self.params = params
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.contexts["deer"] = context.SharedContext(comm) # https://repast.github.io/repast4py.site/guide/user_guide.html#_contexts_and_projections
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['sim']['end_tick'])
        self.runner.schedule_end_event(self.end_sim)

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
        

        # Initialise agents


    def start_sim(self):
        '''
        Start the agents and model.  
        '''
        pylog.debug('Starting model...')
        self.runner.execute()

    def end_sim(self):
        '''
        Clean up and log data.
        '''
        pylog.debug(f'Cleaning up simulation...')

    def add_agent(self):
        '''
        Add and agent to the sim
        '''
        pylog.debug(f'Adding agent...')

    def remove_agent(self):
        '''
        Remove an agent from the sim. 
        '''
        pylog.debug(f'Removing agent...')

    def move_agents(self):
        '''
        Agents move in space over one tick
        '''
        pylog.debug(f'Moving agents...')

    def step(self):
        '''
        Increment the sim by one step.
        '''
        pylog.debug(f'Stepping...')
    
    def log_metrics(self):
        '''
        Log the outputs of the tick to a file.
        '''
        pylog.debug(f'Logging metrics...') 
 

def setup_logging(params):
    '''
    Setup the logging for both Repast4Py as well as a normal python logger
    '''
    if params.get('sim',{}).get('environment') == 'hopper':
        # Do not setup a command line logger.
        pass

    elif params.get('sim',{}).get('environment') == 'local':
        # Setup a command line logger.
        loglevel = params['logging']['loglevel']
        pylog.basicConfig(level=loglevel, format='%(asctime)s %(relativeCreated)6d %(threadName)s ::%(levelname)-8s %(message)s')
        pylog.debug('Ready to log!')
        pylog.info(f'Params: {params}')

    else:
        # Do not setup a command line logger.
        print('Unknown Environment...')

def run(params: Dict):
    setup_logging(params)
    model = Model(MPI.COMM_WORLD, params)
    # model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)
    pylog.info('==END==')
