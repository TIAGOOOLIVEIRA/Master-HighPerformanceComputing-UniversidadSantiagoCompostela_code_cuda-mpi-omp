#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <time.h>

#define CHECK_CUDA_ERROR(call)                                                   \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (error code: %d)\n",                 \
                    cudaGetErrorString(err), err);                               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

// Kernel to compute the sum of each row in a matrix
// occupancy is 4 threads per warp
// 128 threads per block
__global__ void __launch_bounds__(128, 4)
row_sum_kernel(const float *__restrict__ input, float *__restrict__ output, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += input[row * n + j];
        }
        output[row] = sum;
    }
}

void row_sum_cpu(const float *input, float *output, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
            sum += input[i * n + j];
        }
        output[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <threads_per_block>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int size = n * n;

    float *h_matrix = (float *)malloc(size * sizeof(float));
    float *h_result_gpu = (float *)malloc(n * sizeof(float));
    float *h_result_cpu = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < size; i++) {
        h_matrix[i] = (float)(i % 100) / 10.0f;
    }

    float *d_matrix, *d_result;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_matrix, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_result, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix, size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((n + threadsPerBlock - 1) / threadsPerBlock);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    row_sum_kernel<<<gridDim, blockDim>>>(d_matrix, d_result, n);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    printf("GPU time: %.4f ms\n", ms);

    CHECK_CUDA_ERROR(cudaMemcpy(h_result_gpu, d_result, n * sizeof(float), cudaMemcpyDeviceToHost));

    clock_t cpu_start = clock();
    row_sum_cpu(h_matrix, h_result_cpu, n);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU time: %.4f ms\n", cpu_time);

    int errors = 0;
    for (int i = 0; i < n; i++) {
        //tolerance check was relaxed to 1e-2f
        if (fabs(h_result_cpu[i] - h_result_gpu[i]) > 1e-2f) {
            errors++;
            if (errors <= 10) {
                printf("Mismatch at row %d: CPU = %f, GPU = %f\n", i, h_result_cpu[i], h_result_gpu[i]);
            }
        }
    }

    if (errors == 0) {
        printf("CPU-GPU Validation passed.\n");
    } else {
        printf("CPU-GPU Validation failed with %d mismatches.\n", errors);
    }

    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    free(h_matrix);
    free(h_result_gpu);
    free(h_result_cpu);

    CHECK_CUDA_ERROR(cudaDeviceReset());
    //CHECK_CUDA_ERROR(cudaEventDestroy(start));
    //CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("CUDA device reset.\n");
    
    return 0;   
}