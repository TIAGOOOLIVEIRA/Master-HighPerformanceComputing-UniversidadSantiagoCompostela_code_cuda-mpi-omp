#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J oclHeterogDevicesProc       # Job name
#SBATCH -o oclHeterogDevicesProc.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e oclHeterogDevicesProc.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:25:00 #(25 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#module --ignore-cache avail
module load cesga/2020 cuda-samples/11.2

# Run the CUDA program
#OMP_NUM_THREADS=8 ./oclHeterogDevicesProc

# Array of GPU types to iterate over
gpu_types=("t4" "a100")

# Iterate over GPU types and n values
for gpu in "${gpu_types[@]}"; do
    export SLURM_GPUS_ON_NODE=$gpu  # Dynamically set GPU type for execution
    echo "Running GPU=$gpu"
    OMP_NUM_THREADS=8 ./oclHeterogDevicesProc
done