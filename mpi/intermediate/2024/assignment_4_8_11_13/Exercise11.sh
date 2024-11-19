#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J run_smoothing_grayscale       # Job name
#SBATCH -o run_smoothing_grayscale.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e run_smoothing_grayscale.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:25:00 #(25 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=9

module load intel impi

OMP_NUM_THREADS=8 mpirun -np $SLURM_NTASKS ./smoothing_grayscale