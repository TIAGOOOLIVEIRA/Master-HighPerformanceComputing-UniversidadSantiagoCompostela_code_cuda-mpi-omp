#!/bin/bash
#SBATCH -J row_sum
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH -c 32
#SBATCH --gres=gpu:a100
#SBATCH --mem=64GB

#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x mergesort.sh
#module avail cuda

#to run in interactive mode
#compute --gpu
#./mergesort.sh
####salloc -I600  --qos=viz -p viz --gres=gpu:t4 --mem=3952M -c 1 -t 08:00:00  srun -c 1 --pty --preserve-env /bin/bash -i


module load cesga/2020 
module load cuda-samples/12.2
module load cuda/12.2.0

echo "Compiling with profiling support"
#nvcc -arch=sm_70 -O3 -Xcompiler="-march=native -fopenmp" mergesort.cu -o mergesort
make 
make profile

echo "setting OMP_NUM_THREADS to 8"
export OMP_NUM_THREADS=8 

echo "Running: ./mergesort"
./mergesort