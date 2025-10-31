#!/bin/bash --login
#SBATCH --job-name=nitAI_YYYY_
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000M
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --distribution=block:block
#SBATCH --partition=mod

#Question: How many cpus per node?

# Set the number of threads to 1 as required by all subjobs
export OMP_NUM_THREADS=1

#@Deep:: WARNING!! This code pred_NO3_optm_1core.py is not meant to run on parallel processors run it with one core always.

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

CURR_DIR=$(pwd)
year=_YYYY_
srun --nodes=1 --ntasks=1 python ${CURR_DIR}/pred_NO3_optm_1core.py ${year} -s 'Test 1' &
wait
