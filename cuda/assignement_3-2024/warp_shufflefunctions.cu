#include <stdio.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

/*
To compile
    compute --gpu
    nvcc -g -G -o warp_shufflefunctions warp_shufflefunctions.cu -Xcompiler -fopenmp

To submit and wath the job:
  sbatch CR_cuda.sh
  watch squeue -u curso370 

*/


#define EPSILON 1e-5

#define CHECK_CUDA_ERROR(call)                                                   \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (error code: %d)\n",                 \
                    cudaGetErrorString(err), err);                               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

__global__ void combined_step1_step2_kernel(const float *__restrict__ input, float *__restrict__ result, int rows, int cols, int depth, int P);
__global__ void step3_kernel(const float *__restrict__ input, float *__restrict__ result, int rows, int cols, int depth);


bool validate_results(float *cpu_result, float *gpu_result, int rows, int cols, int depth) {
    bool valid = true;

    #pragma omp parallel for collapse(3) shared(valid)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int l = 0; l < depth; l++) {
                int idx = i * cols * depth + j * depth + l;
                if (fabs(cpu_result[idx] - gpu_result[idx]) > EPSILON) {
                    #pragma omp critical
                    {
                        valid = false;
                        printf("Mismatch at (%d, %d, %d): CPU=%f, GPU=%f\n",
                               i, j, l, cpu_result[idx], gpu_result[idx]);
                    }
                }
            }
        }
    }

    return valid;
}

void combined_step1_step2_cpu(float *input, float *result, int rows, int cols, int depth, int P) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int l = 0; l < depth; l++) {
                int idx = i * cols * depth + j * depth + l;

                // Step 1: Calculate the minimum over P values
                float min_value = input[idx];
                for (int k = 1; k <= P && (l + k) < depth; k++) {
                    int k_idx = i * cols * depth + j * depth + (l + k);
                    min_value = fminf(min_value, input[k_idx]);
                }

                // Step 2: Add left and right neighbors
                float left = (j > 0) ? input[i * cols * depth + (j - 1) * depth + l] : 0.0f;
                float right = (j < cols - 1) ? input[i * cols * depth + (j + 1) * depth + l] : 0.0f;

                // Final result
                result[idx] = input[idx] + min_value + left + right;
            }
        }
    }
}


void step3_cpu(float *input, float *result, int rows, int cols, int depth) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int l = 0; l < depth; l++) {
                int idx = i * cols * depth + j * depth + l;

                // Access top and bottom neighbors in the row dimension
                float top = (i > 0) ? input[idx - cols * depth] : 0.0f;
                float bottom = (i < rows - 1) ? input[idx + cols * depth] : 0.0f;

                // Final result
                result[idx] = input[idx] + top + bottom;
            }
        }
    }
}

// Main function
int main() {
    int rows = 5, cols = 5, depth = 32, P = 3; // Example parameters
    size_t size = rows * cols * depth * sizeof(float);

    // Allocate memory on host and device
    float *h_input = (float *)malloc(size);
    float *h_result = (float *)malloc(size);
    float *h_final_result = (float *)malloc(size);
    float *d_input, *d_intermediate, *d_result;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_intermediate, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, size));

    CHECK_CUDA_ERROR(cudaMemset(d_result, 1, size));


    // Initialize input matrix
    for (int i = 0; i < rows * cols * depth; i++) {
        h_input[i] = (float)(i % 10);
    }

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(32, 1, 1); // Warp size is 32
    dim3 blocksPerGrid(rows, cols, 1);

    // Shared memory size for combined kernel
    size_t sharedMemorySize = cols * sizeof(float);

    // Launch combined kernel for Steps 1 and 2
    combined_step1_step2_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(
        d_input, d_intermediate, rows, cols, depth, P
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Launch Step 3 kernel
    step3_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_intermediate, d_result, rows, cols, depth);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy final result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    // Validate the GPU results
    // Compute the CPU version for validation
    float *h_cpu_result = (float *)malloc(size);
    combined_step1_step2_cpu(h_input, h_cpu_result, rows, cols, depth, P);
    step3_cpu(h_cpu_result, h_final_result, rows, cols, depth);

    // Validate
    if (validate_results(h_final_result, h_result, rows, cols, depth)) {
        printf("Validation PASSED: GPU results match CPU results.\n");
    } else {
        printf("Validation FAILED: GPU results do not match CPU results.\n");
    }

    // Free memory
    free(h_input);
    free(h_result);
    free(h_final_result);
    free(h_cpu_result);
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_intermediate));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}


__global__ void combined_step1_step2_kernel(const float *__restrict__ input, float *__restrict__ result, int rows, int cols, int depth, int P) {

    extern __shared__ float shared[];

    int i = blockIdx.x;
    int j = blockIdx.y;
    int l = threadIdx.x;
    int warpIdx = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    if (i < rows && j < cols && l < depth) {
        int idx = i * cols * depth + j * depth + l;

        // Step 1: Warp-level minimum calculation
        float value = input[idx];
        float min_value = value;

        // Calculate the minimum over P values in the depth dimension using warp shuffle
        for (int k = 1; k <= P; k++) {
            if (l + k < depth) {
                int k_idx = i * cols * depth + j * depth + (l + k);
                min_value = fminf(min_value, input[k_idx]);
            }
        }

        //Reduce minimum value across the warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            min_value = fminf(min_value, __shfl_down_sync(0xffffffff, min_value, offset));
            printf("Thread (%d, %d, %d): min_value=%f\n", i, j, l, min_value);
        }

        //Warp result in shared memory
        if (lane == 0) {
            shared[warpIdx] = min_value;
        }
        __syncthreads();

        // Step 2: Read from shared memory and compute the result
        float left = (j > 0) ? shared[(j - 1) * blockDim.y + warpIdx] : 0.0f;
        float right = (j < cols - 1) ? shared[(j + 1) * blockDim.y + warpIdx] : 0.0f;

        printf("Thread (%d, %d, %d): input=%f, result=%f\n", i, j, l, input[idx], result[idx]);

        // Final result for Step 2
        result[idx] = value + shared[warpIdx] + left + right;
    }
}

__global__ void step3_kernel(const float *__restrict__ input, float *__restrict__ result, int rows, int cols, int depth) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int l = threadIdx.z;

    if (i < rows && j < cols && l < depth) {
        int idx = i * cols * depth + j * depth + l;

        //Top and bottom neighbors in the row dimension
        float top = (i > 0) ? input[idx - cols * depth] : 0.0f;       // (i-1)
        float bottom = (i < rows - 1) ? input[idx + cols * depth] : 0.0f; // (i+1)

        // Final result for Step 3
        result[idx] = input[idx] + top + bottom;
    }
}
