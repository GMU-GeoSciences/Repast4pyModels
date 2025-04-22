#!/bin/bash

#Submit this script with: sbatch singularity_run.slurm

#SBATCH --job-name=deer_model   ##name that will show up in the queue
#SBATCH --output=/home/%u/Repast4pyModels/output/%x-%N-%j.out   ##filename of the output
#SBATCH --error=/home/%u/Repast4pyModels/output/%x-%N-%j.err   ##filename of the error output
#SBATCH --nodes=2  ##number of nodes to use
#SBATCH --ntasks=80  ##number of tasks to run
#SBATCH --time=0-2:00:00  ##time for analysis (day-hour:min:sec)
#SBATCH --cpus-per-task=1  ##the number of threads the code will use
#SBATCH --mem=20GB  ##memory allocation per node
#SBATCH --partition normal  ##the partition to run in
#SBATCH --mail-user rmeyer6@gmu.edu  ##your email address
#SBATCH --mail-type BEGIN  ##slurm will email you when your job starts
#SBATCH --mail-type END  ##slurm will email you when your job ends
#SBATCH --mail-type FAIL  ##slurm will email you when your job fails


## Commands to Load Modules:
module load gnu9 intel mpich git singularity


## Load modules, insert code, and run your programs here.
mpirun -n 80 singularity run /containers/hopper/UserContainers/$USER/repast4py_latest.sif python ./repast4py/deer_model.py ./config/hopper_deer_config.yaml
