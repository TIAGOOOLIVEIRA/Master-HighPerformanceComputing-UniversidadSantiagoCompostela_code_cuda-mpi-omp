#!/bin/bash
#SBATCH -J pi_integral
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH -c 32
#SBATCH --gres=gpu:t4
#SBATCH --mem=64GB

#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x pi_integral.sh
#module avail cuda

#to run in interactive mode
#compute --gpu
#./pi_integral.sh

module load gcc openmpi/4.0.5_ft3
module load intel vtune
module load intel impi

echo "Compiling with profiling support"
make clean
make

echo "setting OMP_NUM_THREADS to 8"
export OMP_NUM_THREADS=8

#for averaging
runs=3

if [ $? -ne 0 ]; then
    echo "Row sum matrices CUDA app Compilation failed!"
    exit 1
fi

#Matrix sizes and block sizes to test. Ideally to run over two different GPUs
matrices=(8 16 32)
matrix_size=(128 256 512)
#gpu_types=("t4" "a100")
gpu_types=("t4")

#Execute over all configs to collect stats for report
for gpu in "${gpu_types[@]}"; do
    echo "=== Executing on GPU: $gpu ==="
    nvidia-smi
    #export SLURM_GPUS_ON_NODE=$gpu
    for nmatrices in "${matrices[@]}"; do
        for msize in "${matrix_size[@]}"; do
            for ((r=1; r<=runs; r++)); do
                echo "Run $r for size $nmatrices $msize"
                echo "Running: ./pi_integral $msize $nmatrices on GPU $gpu"
                OMP_NUM_THREADS=8 ./pi_integral $msize $nmatrices
            done
        done
    done
done

echo "Execution completed."

echo "Profiling with nsys, vtune(CPU) and ncu"
make profile
make vtune-cpu
make ncu