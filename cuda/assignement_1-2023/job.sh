#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run CUDA applications
# on CESGA's FT-III system.
#----------------------------------------------------
#SBATCH -J gpu_job       # Job name
#SBATCH -o gpu_job.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e gpu_job.o%j   # Name of stderr output file(%j expands to jobId)
#SBATCH -c 32            # Cores per task requested (1 task job)
# Needed 32 cores per A100 demanded
#SBATCH --mem-per-cpu=3G # memory per core demanded
#SBATCH --gres=gpu       # Options for requesting 1GPU
#SBATCH -t 01:30:00      # Run time (hh:mm:ss) - 1.5 hours

# Run the CUDA application
#./euclideanvectornorm 2000 4000 32
#./euclideanvectornorm 2000 4000 64
#./euclideanvectornorm 2000 4000 128
#./euclideanvectornorm 4000 2000 32
#./euclideanvectornorm 4000 2000 64
#./euclideanvectornorm 4000 2000 128
#./euclideanvectornorm 10000 40000 32
#./euclideanvectornorm 10000 40000 64
#./euclideanvectornorm 10000 40000 128
#./euclideanvectornorm 40000 10000 32
#./euclideanvectornorm 40000 10000 64
#./euclideanvectornorm 40000 10000 128

module load cesga/2020 cuda-samples/11.2
./euclideanvectornorm 2000 4000 32

#sbatch job.sh
#watch -n 1 squeue -u curso370