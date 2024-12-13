# Makefile for the repast4py project.
#####################################
# This is to quickly test and start sims on either
# a dev environment or the Hopper cluster at GMU.
DEV_NAME=$(shell ./make-scripts/where_am_i.sh)

env:
ifeq ($(DEV_NAME), hopper)
	@echo =================================
	@echo == Running in HPC environment! ==
	@echo =================================
 
else ifeq ($(DEV_NAME), dev)
	@echo =================================
	@echo == Running in dev environment! ==
	@echo =================================
	
else
	@echo Running on unknown machine: $(DEV_NAME)
endif

# Setup environment based on hostname. Checks if .build file exists
# before rebuilding
build: env
ifeq ($(DEV_NAME), hopper)
	./make-scripts/hopper_singularity_setup.sh
else
	./make-scripts/local_setup.sh
endif
	touch build

# Run Deer model based on hostname
deer_run: env build
ifeq ($(DEV_NAME), hopper)
	sbatch ./make-scripts/slurm_scripts/singularity_run.slurm
else
	./make-scripts/local_run.sh
endif

# Run a benchmark using a specific config file
benchmark: env build
#TODO: Write benchmark script + config file
ifeq ($(DEV_NAME), hopper)
	./make-scripts/hopper_singularity_benchmark.sh
else
	./make-scripts/local_benchmark.sh
endif
	rm build

# Clean up everything; remove docker image, remove raster files
clean: env
ifeq ($(DEV_NAME), hopper)
	./make-scripts/hopper_cleanup.sh
else
	./make-scripts/local_cleanup.sh
endif
	rm build

