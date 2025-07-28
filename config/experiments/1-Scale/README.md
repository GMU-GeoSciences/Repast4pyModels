# Experiment 1: Does this scale? 

This experiment aims to determine how well this library scales with dear agents. To properly measure this it should be run on Hopper and not locally. The three config files use the same movement models, and location info, and time parameters but have increasing number of agents.

The goal is to measure the impact that increasing the number of agents has on the performance, and when does it start getting too slow for real world use. 

Steps to run:

'''
# On Slurm
> git pull
> make build
> make experiment1
'''

Each config file in Repast4pyModels/config/experiments/1-Scale/. is used to run a different Slurm batch script and has an increasing number of agents. 

The results are saved to a csv file, but more importantly the meta data for each run (duration, number of agents etc) is saved to a json file. This is then used to measure performance in a Jupyter notebook.

