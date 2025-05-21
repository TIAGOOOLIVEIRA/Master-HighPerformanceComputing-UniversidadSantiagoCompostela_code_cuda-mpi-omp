#include "shared/helper.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <time.h>
#include <math.h>

__global__ void row_sum_kernel(const float *__restrict__ input, float *__restrict__ output, int n) {
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

void validate_cpu_gpu(float **cpu_results, float **gpu_results, int n, int m) {
    for (int i = 0; i < m; i++) {
        int errors = 0;
        for (int j = 0; j < n; j++) {
            if (fabs(cpu_results[i][j] - gpu_results[i][j]) > 1e-2f) {
                errors++;
                if (errors <= 10) {
                    printf("Mismatch at matrix %d, row %d: CPU = %f, GPU = %f\n", i, j, cpu_results[i][j], gpu_results[i][j]);
                }
            }
        }
        if (errors > 0) {
            printf("Validation failed for matrix %d with %d mismatches.\n", i, errors);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <num_matrices>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int size = n * n;

    float **h_inputs = (float **)malloc(m * sizeof(float *));
    float **h_results_gpu = (float **)malloc(m * sizeof(float *));
    float **h_results_cpu = (float **)malloc(m * sizeof(float *));

    for (int i = 0; i < m; i++) {
        h_inputs[i] = (float *)malloc(size * sizeof(float));
        h_results_gpu[i] = (float *)malloc(n * sizeof(float));
        h_results_cpu[i] = (float *)malloc(n * sizeof(float));

        for (int j = 0; j < size; j++) {
            h_inputs[i][j] = (float)((i * size + j) % 100) / 10.0f;
        }
    }

    float *d_input[m], *d_output[m];
    cudaStream_t streams[m];

    for (int i = 0; i < m; i++) {
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input[i], size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output[i], n * sizeof(float)));
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    Timer timer;
    timer.start();

    for (int i = 0; i < m; i++) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input[i], h_inputs[i], size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        dim3 blockDim(128);
        dim3 gridDim((n + 127) / 128);
        row_sum_kernel<<<gridDim, blockDim, 0, streams[i]>>>(d_input[i], d_output[i], n);
        //CHECK_CUDA_ERROR(cudaMemcpyAsync(h_results_gpu[i], d_output[i], n * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    for (int i = 0; i < m; i++) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_results_gpu[i], d_output[i], n * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        //CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer.stop("GPU time for %d matrices");

    CpuTimer cpu_timer;
    cpu_timer.start();
    for (int i = 0; i < m; i++) {
        row_sum_cpu(h_inputs[i], h_results_cpu[i], n);
    }
    cpu_timer.stop("CPU time for matrices");

    validate_cpu_gpu(h_results_cpu, h_results_gpu, n, m);

    for (int i = 0; i < m; i++) {
        CHECK_CUDA_ERROR(cudaFree(d_input[i]));
        CHECK_CUDA_ERROR(cudaFree(d_output[i]));
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        free(h_inputs[i]);
        free(h_results_gpu[i]);
        free(h_results_cpu[i]);
    }

    CHECK_CUDA_ERROR(cudaDeviceReset());
    printf("CUDA device reset.\n");
    return 0;
}
