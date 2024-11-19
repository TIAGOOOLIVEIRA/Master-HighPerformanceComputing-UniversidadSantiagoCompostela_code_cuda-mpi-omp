#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J CR_cuda       # Job name
#SBATCH -o CR_cuda.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e CR_cuda.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:25:00 #(25 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#module --ignore-cache avail
module load cesga/2020 cuda-samples/11.2

# Run the CUDA program
#OMP_NUM_THREADS=8 ./CR_cuda

n_values=(8 16 32 64 128 256 512 1024 2048 4096)

# Array of GPU types to iterate over
gpu_types=("t4" "a100")

# Iterate over GPU types and n values
for gpu in "${gpu_types[@]}"; do
    export SLURM_GPUS_ON_NODE=$gpu  # Dynamically set GPU type for execution
    for n in "${n_values[@]}"; do
        B=$((2**24 / n)) # Calculate B as 2^24 / n
        echo "Running CR_cuda with n=$n, B=$B, and GPU=$gpu"
        OMP_NUM_THREADS=8 ./CR_cuda $n
    done
done