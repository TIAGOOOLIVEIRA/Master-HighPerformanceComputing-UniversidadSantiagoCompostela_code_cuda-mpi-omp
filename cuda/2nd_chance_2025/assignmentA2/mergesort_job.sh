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
#chmod +x mergesort_job.sh
#module avail cuda

#to run in interactive mode
#compute --gpu
#./mergesort.sh
####salloc -I600  --qos=viz -p viz --gres=gpu:t4 --mem=3952M -c 1 -t 08:00:00  srun -c 1 --pty --preserve-env /bin/bash -i


module load cesga/2020 
module load cuda-samples/12.2
module load cuda/12.2.0
module load intel vtune

echo "Compiling with profiling support"
#nvcc -arch=sm_70 -O3 -Xcompiler="-march=native -fopenmp" mergesort_opt.cu -o mergesort_opt
#to also help the bank conflicts analysis
#nsys profile --stats=true --force-overwrite=true -o mergesort-report ./mergesort
#ncu --set=shared_memory --kernel-name "sort_kernel" ./mergesort
#ncu --set full --target-processes all -o mergesort_ncu_t4 ./mergesort
#ncu --target-processes all --launch-count 1 --kernel-name "sort_kernel" --set memory-workload-analysis -o mergesort_ncu_t4 ./mergesort



make 


echo "setting OMP_NUM_THREADS to 8"
export OMP_NUM_THREADS=8 

#Array sizes
sizes=(128 256 512 1024 2048 4096 8192 16384)

#for averaging
runs=3

#Output CSV - analysis
echo "ArraySize,Run,GPUBatchSort(ms),CPUBatchSort(ms)" > benchmark_results.csv

for size in "${sizes[@]}"; do
    echo "=============================="
    echo "Benchmarking array size: $size"
    echo "=============================="

    for ((r=1; r<=runs; r++)); do
        echo "Run $r for size $size"

        OUTPUT=$(./mergesort $size)

        GPU_TIME=$(echo "$OUTPUT" | grep "GPU TIMING \[GPU Batch Merge Sort\]" | awk '{print $6}')
        CPU_TIME=$(echo "$OUTPUT" | grep "GPU TIMING \[CPU Batch Merge Sort\]" | awk '{print $6}')


        echo "$size,$r,$GPU_TIME,$CPU_TIME" >> benchmark_results.csv

    done
done

echo "Execution completed. Results saved to benchmark_results.csv."

echo "Profiling with nsys and ncu"
make profile
make vtune-cpu
#make ncu