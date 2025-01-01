#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                                   \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (error code: %d)\n",                 \
                    cudaGetErrorString(err), err);                               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

int main() {
    int rows = 5, cols = 5, depth = 3, P = 2;
    size_t size = rows * cols * depth * sizeof(float);

    // Allocate memory on host and device
    float *h_input = (float *)malloc(size);
    float *h_result = (float *)malloc(size);
    float *d_input, *d_result;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, size));

    // Initialize input matrix
    for (int i = 0; i < rows * cols * depth; i++) {
        h_input[i] = (float)(i + 1); // Example initialization
    }

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((rows + 15) / 16, (cols + 15) / 16, depth);

    // Step 1
    step1_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_result, rows, cols, depth, P);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Step 2
    step2_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_input, rows, cols, depth);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Step 3
    step3_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_result, rows, cols, depth);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost));

    // Print results
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int l = 0; l < depth; l++) {
                int idx = i * cols * depth + j * depth + l;
                printf("%f ", h_result[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Free memory
    free(h_input);
    free(h_result);
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
