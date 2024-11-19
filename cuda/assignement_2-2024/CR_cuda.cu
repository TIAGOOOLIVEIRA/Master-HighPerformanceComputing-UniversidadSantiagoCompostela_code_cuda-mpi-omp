/*
 *Heterogeneous Programming
 Assingment 2
 Student: Tiago de Souza Oliveira

https://cesga-docs.gitlab.io/ft3-user-guide/gpu_nodes.html#nvidia-a100
https://cesga-docs.gitlab.io/ft3-user-guide/batch_examples.html#using-sbatch-and-gpus

To compile
    compute --gpu
    nvcc -o CR_cuda CR_cuda.cu  -Xcompiler -fopenmp -lcudart


To submit and wath the job:
  sbatch CR_cuda.sh
  watch squeue -u curso370 

This job is executed from a shell script "Exercise08.sh" composed by the following instructions:

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

*/


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

const int N = 16;

#include <sys/time.h>
#include <sys/resource.h>

#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo {
  double time;
  double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ",		\
			    t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

void myElapsedtime(const resnfo start, const resnfo end, timenfo *const t)
{
#ifdef _noWALL_
  t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6)) 
    - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
  t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6)) 
    - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
#else
  *t = (end.tv_sec + (end.tv_usec * 1E-6)) 
    - (start.tv_sec + (start.tv_usec * 1E-6));
#endif /*_noWALL_*/
}

__global__ void CR_GPU(float *A, float *B, float *C, float *D, float *X, int n) {
    __shared__ float s_A[N];
    __shared__ float s_B[N];
    __shared__ float s_C[N];
    __shared__ float s_D[N];

    int tid = threadIdx.x;
    int block_id = blockIdx.x;

    s_A[tid] = A[block_id * n + tid];
    s_B[tid] = B[block_id * n + tid];
    s_C[tid] = C[block_id * n + tid];
    s_D[tid] = D[block_id * n + tid];
    __syncthreads();

    int stride = 2, step = 1, k = n - 1;
    float s1, s2;
    for (int i = 0; i < log2f((float)n) - 1; i++) {
        if (tid >= step && tid < n - 1 && tid % stride == 0) {
            s1 = s_A[tid] / s_B[tid - step];
            s2 = s_C[tid] / s_B[tid + step];

            s_A[tid] = -s_A[tid - step] * s1;
            s_B[tid] = s_B[tid] - s_C[tid - step] * s1 - s_A[tid + step] * s2;
            s_C[tid] = -s_C[tid + step] * s2;
            s_D[tid] = s_D[tid] - s_D[tid - step] * s1 - s_D[tid + step] * s2;
        }

        if (tid == k) {
            s1 = s_A[k] / s_B[k - step];
            s_A[k] = -s_A[k - step] * s1;
            s_B[k] = s_B[k] - s_C[k - step] * s1;
            s_D[k] = s_D[k] - s_D[k - step] * s1;
        }
        step += stride;
        stride *= 2;
        __syncthreads();
    }

    if (tid == n / 2 - 1 || tid == n - 1) {
        k = n / 2 - 1;
        int l = n - 1;
        float denominator = s_B[k] * s_B[l] - s_C[k] * s_A[l];
        X[block_id * n + k] = (s_B[l] * s_D[k] - s_C[k] * s_D[l]) / denominator;
        X[block_id * n + l] = (s_D[l] * s_B[k] - s_D[k] * s_A[l]) / denominator;
    }
    __syncthreads();

    int step_back = n / 4, stride_back = n / 2, idx = step_back - 1;
    for (int i = 0; i < log2f((float)n) - 1; i++) {
        if (tid == idx) {
            X[block_id * n + idx] = (s_D[idx] - s_C[idx] * X[block_id * n + idx + step_back]) / s_B[idx];
        }
        if (tid >= idx + stride_back && tid < n && tid % stride_back == 0) {
            X[block_id * n + tid] = (s_D[tid] - s_A[tid] * X[block_id * n + tid - step_back] - s_C[tid] * X[block_id * n + tid + step_back]) / s_B[tid];
        }
        step_back /= 2;
        stride_back /= 2;
        idx = step_back - 1;
        __syncthreads();
    }
}

void Initialization(float *A, float *B, float *C, float *D, int n, int num_systems) {
    #pragma omp parallel for
    for (int i = 0; i < num_systems; i++) {
        A[i * n] = 0.0;
        B[i * n] = 2.0;
        C[i * n] = -1.0;
        D[i * n] = 1.0;

        #pragma omp parallel for
        for (int j = 1; j < n - 1; j++) {
            A[i * n + j] = -1.0;
            B[i * n + j] = 2.0;
            C[i * n + j] = -1.0;
            D[i * n + j] = 0.0;
        }

        A[i * n + n - 1] = -1.0;
        B[i * n + n - 1] = 2.0;
        C[i * n + n - 1] = 0.0;
        D[i * n + n - 1] = 1.0;
    }
}

int main(int argc, char *argv[]){
    // Número de elementos en los vectores (predeterminado: N)
    unsigned int n = (argc > 1)?atoi (argv[1]):N;

    unsigned int num_systems = (1 << 24) / n;

    resnfo start, end;
    timenfo time;


    int numBytes = n * num_systems * sizeof(float);
    float *h_A = (float *)malloc(numBytes);
    float *h_B = (float *)malloc(numBytes);
    float *h_C = (float *)malloc(numBytes);
    float *h_D = (float *)malloc(numBytes);
    float *h_X = (float *)malloc(numBytes);

    Initialization(h_A, h_B, h_C, h_D, n, num_systems);
    
    float *d_A, *d_B, *d_C, *d_D, *d_X;
    cudaMalloc(&d_A, numBytes);
    cudaMalloc(&d_B, numBytes);
    cudaMalloc(&d_C, numBytes);
    cudaMalloc(&d_D, numBytes);
    cudaMalloc(&d_X, numBytes);

    cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, numBytes, cudaMemcpyHostToDevice);

    dim3 blockDim(n);
    dim3 gridDim(num_systems);

    timestamp(&start);
    CR_GPU<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D, d_X, n);
    timestamp(&end);

    cudaMemcpy(h_X, d_X, numBytes, cudaMemcpyDeviceToHost);

    myElapsedtime(start, end, &time);
    printtime(time);
    printf(" -> CR en la  GPU  \n");

    //Shared memory for A, B, C, D
    unsigned int shared_mem_size = 4 * n * sizeof(float); 
    printf("Shared memory per block: %d bytes\n", shared_mem_size);
    printf("Threads per block: %d\n", blockDim.x);
    printf("Number of blocks: %d\n\n", gridDim.x);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_X);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_X);

    cudaDeviceReset();

    return 0;
}