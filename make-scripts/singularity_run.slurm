#!/bin/bash
#SBATCH   --partition=normal            # submit   to the normal(default) partition
#SBATCH   --job-name=deer_repast             # name the job
#SBATCH   --output=deer_repast-%j.out        # write stdout/stderr   to named file
#SBATCH   --error=deer_repast-%j.err      
#SBATCH   --time=0-02:00:00             # Run for max of 02 hrs, 00 mins, 00 secs
#SBATCH   --nodes=1                     # Request N nodes
#SBATCH   --cpus-per-task=10            # Request n   cores per node
#SBATCH   --mem-per-cpu=2GB             # Request nGB RAM per core


set echo 
umask 0022 
nvidia-smi 
env|grep -i slurm

SINGULARITY_BASE=/containers/hopper/UserContainers/$USER/
CONTAINER=${SINGULARITY_BASE}/repast4py_latest.sif
SINGULARITY_RUN="singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd" 

SCRIPT=./repast4py/deer_model.py
ARGS=./config/hopper_deer_config.yaml

${SINGULARITY_RUN} ${CONTAINER} python ${SCRIPT} ${SCRIPT} | tee output/slurm.log