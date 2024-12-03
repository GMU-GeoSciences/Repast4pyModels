# Makefile for the repast4py project.
#####################################
# This is to quickly test and start sims on either
# a dev environment or the Hopper cluster at GMU.

HOST=$(shell hostname) #Get name of machine where this is running
HPC_NAME=hopper1.orc.gmu.edu 

# Setup environment based on hostname. Checks if .build file exists
# before rebuilding
build: 
ifeq ($(HOST), $(HPC_NAME))
	./make-scripts/hopper_singularity_setup.sh
else
	./make-scripts/local_setup.sh
endif
	touch build

# Run Deer model based on hostname
deer_run: build 
ifeq ($(HOST), $(HPC_NAME))
	./make-scripts/hopper_singularity_run.sh
else
	./make-scripts/local_run.sh
endif

# Run a benchmark using a specific config file
benchmark: build
#TODO: Write benchmark script + config file

# Clean up everything; remove docker image, remove raster files
clean:
ifeq ($(HOST), $(HPC_NAME))
	./make-scripts/hopper_cleanup.sh
else
	./make-scripts/local_cleanup.sh
endif
	rm build

all:
ifeq ($(HOST), $(HPC_NAME))
	@echo Running on HPC.
else ifeq ($(HOST), rorybox)
	@echo Running on Rory\'s Laptop
else
	@echo Running on $(HOST)
endif
