# Repast4pyModels

This is a repository containing scripts to setup the Repast4py environment in GMU's Hopper high performance cluster. The goal being to run several distributed agent based models.

## Sub Project: Deer Covid modelling

Covid-19 has been detected in a local deer population. This project aims to combine deer behaviour models and human population models in order to run scenarios of human <> deer infections.

## Quick Start

### Local Machine
This is when you want to run this on your local machine, to test or develop further.

```bash
git clone https://github.com/GMU-GeoSciences/Repast4pyModels.git 
cd Repast4pyModels
make build
make deer_run
```


This guide is for GMU members who have access to the Slurm/Hopper cluster.

### HPC Cluster
Here's how you run this on Hopper:

- [Log into Hopper](https://wiki.orc.gmu.edu/mkdocs/Logging_Into_Hopper/)
- Load Git module, pull repo, and run env code:

```bash
module load git
git clone https://github.com/GMU-GeoSciences/Repast4pyModels.git 
cd Repast4pyModels
make build
make deer_run
```

## Other ReadMe's
There are additional readme's in this repo to futher explain the code and it's functionality.

- [Config](./config/README.md)
- [MakeFiles](./make-scripts/README.md)
- [Notebooks](./Notebooks/README.md)
- [Input Files](./input/README.md)
- [Repast Models](./repast4py/README.md)


## How to Contribute

Here's how you contribute to this project:

- Create a branch
- Add some code
- Request that your branch get merged into main

## Containers

Hopper can use singularity containers to run code. It also looks like it's not too difficult to convert Docker containers to Singularity containers. The best place to start seems to be with a Nvidia GPU optimised container and add code to there (even if you're not going to use the GPU)

- [Hopper Singularity README](https://wiki.orc.gmu.edu/mkdocs/Containerized_jobs_on_Hopper/)
- [Nvidia Python Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/distroless/containers/python/tags)
- [NASA guide on converting Docker > Singularity](https://www.nas.nasa.gov/hecc/support/kb/converting-docker-images-to-singularity-for-use-on-pleiades_643.html)

## Additional Reading

[Makefile tutorial](https://makefiletutorial.com/#the-essence-of-make)
[Deer Landscape Disease Model](https://www.researchgate.net/publication/363077733_The_effect_of_landscape_transmission_mode_and_social_behavior_on_disease_transmission_Simulating_the_transmission_of_chronic_wasting_disease_in_white-tailed_deer_Odocoileus_virginianus_populations_usi)
