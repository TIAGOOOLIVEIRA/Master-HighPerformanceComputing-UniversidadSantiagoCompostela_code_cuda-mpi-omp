#!/bin/bash
#SBATCH -J pipeline_demo
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH -c 32
#SBATCH --gres=gpu:t4
#SBATCH --mem=64GB

#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x pipeline_demo.sh
#module avail cuda

#to run in interactive mode
#compute --gpu
#./pipeline_demo.sh

module load cesga/2020 
module load cuda-samples/12.2
module load cuda/12.2.0
module load intel vtune

echo "Compiling with profiling support"
make clean
make

#for averaging
runs=3

if [ $? -ne 0 ]; then
    echo "pipeline_demo CUDA app Compilation failed!"
    exit 1
fi

#Execute over all configs to collect stats for report

echo "=== Executing on GPU: $gpu ==="
nvidia-smi

for ((r=1; r<=runs; r++)); do
    echo "Run $r of $runs"
    echo "Running: ./pipeline_demo"
    ./pipeline_demo
done

echo "Execution completed."

echo "Profiling with nsys, vtune(CPU) and ncu"
make profile
make vtune-cpu
make ncu