#!/bin/bash
#SBATCH -J row_sum
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH -c 32
#SBATCH --gres=gpu:t4
#SBATCH --mem=64GB

#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x row_sum.sh
#module avail cuda

#to run in interactive mode
#compute --gpu
#./row_sum.sh
####salloc -I600  --qos=viz -p viz --gres=gpu:t4 --mem=3952M -c 1 -t 08:00:00  srun -c 1 --pty --preserve-env /bin/bash -i


module load cesga/2020 
module load cuda-samples/12.2
module load cuda/12.2.0


echo "Compiling with -g -G for profiling support"
nvcc -Xcompiler -fopenmp -O2 -g -G -o row_sum row_sum.cu

#to collect profiling data
export OMP_NUM_THREADS=8
nsys profile -o row_sum_nsys ./row_sum 20480 128
ncu --set full --target-processes all -o row_sum_ncu ./row_sum 20480 128

if [ $? -ne 0 ]; then
    echo "Row sum CUDA app Compilation failed!"
    exit 1
fi

#Matrix sizes and block sizes to test. Ideally to run over two different GPUs
matrix_sizes=(5376 10880 20480)
block_sizes=(32 64 128)
#gpu_types=("t4" "a100")
gpu_types=("t4")

#Execute over all configs to collect stats for report
for gpu in "${gpu_types[@]}"; do
    echo "=== Executing on GPU: $gpu ==="
    nvidia-smi
    #export SLURM_GPUS_ON_NODE=$gpu
    for size in "${matrix_sizes[@]}"; do
        for block in "${block_sizes[@]}"; do
            echo "Running: ./row_sum $size $block on GPU $gpu"
            OMP_NUM_THREADS=8 ./row_sum $size $block
        done
    done
done
