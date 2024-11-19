#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J run_exercise11_Reduce_scatter       # Job name
#SBATCH -o run_exercise11_Reduce_scatter.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e run_exercise11_Reduce_scatter.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:10:00 #(10 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=4

module load intel impi

mpirun -np $SLURM_NTASKS ./run_exercise11_Reduce_scatter