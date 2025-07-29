# Scripts

The Repast4pyModels/make-scripts/ subfolder holds scripts to run the models in various ways. In general the simulation is run using:

- Docker: This is for testing and running in a dev environment
- [Singularity](https://wiki.orc.gmu.edu/mkdocs/Containerized_jobs_on_Hopper/): This is for running containerised apps in a HPC environment where Docker is not appropriate.
- Slurm Scheduler: This is for when you would like to provision and queue a job in the HPC environment, and receive updates on the progress of the job.

The generation of Docker/Singularity containers is handled using the [local_setup.sh](./make-scripts/local_setup.sh) and [hopper_setup.sh](./make-scripts/hopper_setup.sh) files. 

The scripts are called by the Makefile in the root directory of this project. The makefile calls the scripts based on whether or not the hostname of the machine is recognised as one of those of the HPC cluster. There might be some of those that are missing...

The Slurm scripts were generated using[ this tool.](https://wiki.orc.gmu.edu/mkdocs/slurm_generator/slurm_script_generator.html)
