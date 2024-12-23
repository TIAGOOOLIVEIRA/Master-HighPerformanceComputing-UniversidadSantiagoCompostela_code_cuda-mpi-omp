#include <stdio.h>
#include <cuda_runtime.h>
#include <omp.h>  // Include OpenMP header

// Macro to check CUDA errors
#define CHECK_CUDA_ERROR(call)                                                   \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s (error code: %d)\n",                 \
                    cudaGetErrorString(err), err);                               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

/*
TODO: topic to explore: 
    cudaMallocManaged
    cudaMemPrefetchAsync
    
To compile, profiling and running:
    compute --gpu
    module load cuda/11.0
    
    export OMP_NUM_THREADS=8 

    nvcc -Xptxas -lineinfo -v multiplematrixscaling_mem_opt_fix.cu -o multiplematrixscaling_mem_opt_fix -Xcompiler -fopenmp
    nvprof --export-profile multiplematrixscaling_mem_opt_fix.prof -f --analysis-metrics ./multiplematrixscaling_mem_opt_fix

    nvprof --output-profile multiplematrixscaling_mem_opt_fix.prof --metrics gld_efficiency,gst_efficiency,achieved_occupancy,warp_execution_efficiency -f --analysis-metrics ./multiplematrixscaling_mem_opt_fix

    nvprof --print-gpu-trace --csv ./multiplematrixscaling_mem_opt_fix > profile.csv


    ./multiplematrixscaling_mem_opt_fix
    
    sbatch multiplematrixscaling_mem_opt_fix.sh
    watch -n 1 squeue -u curso370
*/


// Optimized kernel
__global__ void matrix_scaling_factor_kernel_cuda(
    float * __restrict__ data,  // Input and output array; restrict keyword: avoid aliasing with any other pointer
    unsigned int N,             
    const float factor,         // leveraging const for optimization
    unsigned int repeat
) {
    __shared__ float tile[256];

    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int localIdx = threadIdx.x;

    if (globalIdx < N) {
        tile[localIdx] = data[globalIdx];
    }
    __syncthreads();

    if (globalIdx < N) {
        float val = tile[localIdx];
        for (unsigned int i = 0; i < repeat; i++) {
            val *= factor;
        }
        tile[localIdx] = val;
    }
    __syncthreads();


    if (globalIdx < N) {
        data[globalIdx] = tile[localIdx];
    }
}

int main(int argc, char *argv[]) {
    unsigned int N = 1 << 3; 
    unsigned int nn = N * N;
    unsigned int repeat = 2;
    float factors[] = {0.1, 0.2};
    unsigned int factorsLength = sizeof(factors) / sizeof(factors[0]);
    unsigned int sizeBytes = nn * sizeof(float);

    float *h_matrix, *h_results[factorsLength], *d_matrix[factorsLength];

    // Query device properties
    int device;
    CHECK_CUDA_ERROR(cudaGetDevice(&device));

    int supportsUVA = 0;
    int supportsOverlap = 0;

    // Check for Unified Virtual Addressing (UVA) support
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&supportsUVA, cudaDevAttrUnifiedAddressing, device));
    printf("Device %d supports Unified Virtual Addressing (UVA): %s\n", device, supportsUVA ? "Yes" : "No");

    // Check for GPU Overlap support
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&supportsOverlap, cudaDevAttrConcurrentKernels, device));
    printf("Device %d supports GPU Overlap: %s\n", device, supportsOverlap ? "Yes" : "No");


    CHECK_CUDA_ERROR(cudaMallocHost(&h_matrix, sizeBytes));

    #pragma omp parallel for
    for(unsigned int i = 0; i < N; i++){
        for(unsigned int j = 0; j < N; j++){
            unsigned int idx2d = i * N + j;
            h_matrix[idx2d] = i + j + 1;
        }
    } 

    // Allocate separate host buffers for results
    for (unsigned int i = 0; i < factorsLength; i++) {
        CHECK_CUDA_ERROR(cudaMallocHost(&h_results[i], sizeBytes));
    }

    for (unsigned int i = 0; i < factorsLength; i++) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_matrix[i], sizeBytes));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (nn + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t streams[factorsLength];
    cudaEvent_t startk[factorsLength], endk[factorsLength];

    for (unsigned int i = 0; i < factorsLength; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
        CHECK_CUDA_ERROR(cudaEventCreate(&startk[i]));
        CHECK_CUDA_ERROR(cudaEventCreate(&endk[i]));
    }

    // Launch kernels and memory transfers concurrently
    for (unsigned int i = 0; i < factorsLength; i++) {
        printf("\nLaunching kernel for factor: %f, stream ID: %u", factors[i], i);

        // Asynchronous memory copy from host to device
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            d_matrix[i], h_matrix, sizeBytes, cudaMemcpyHostToDevice, streams[i]));

        // Record event before kernel launch
        CHECK_CUDA_ERROR(cudaEventRecord(startk[i], streams[i]));

        // Launch kernel
        matrix_scaling_factor_kernel_cuda<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
            d_matrix[i], nn, factors[i], repeat
        );

        // Record event after kernel execution
        CHECK_CUDA_ERROR(cudaEventRecord(endk[i], streams[i]));

        // Asynchronous memory copy from device to separate host result buffer
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            h_results[i], d_matrix[i], sizeBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams to finish and collect statistics
    for (unsigned int i = 0; i < factorsLength; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        float elapsedTime = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, startk[i], endk[i]));
        printf("\nKernel for factor: %f; Elapsed time: %f ms", factors[i], elapsedTime);
    }

    #pragma omp parallel for
    for (unsigned int i = 0; i < nn; i++) {
        h_matrix[i] = 0.0f;
        for (unsigned int j = 0; j < factorsLength; j++) {
            h_matrix[i] += h_results[j][i];
        }
    }

    printf("\n\nUpdated Matrix:\n");
    #pragma omp parallel for
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            unsigned int idx2d = i * N + j;
            #pragma omp critical
            printf("%f, ", h_matrix[idx2d]);
        }
        #pragma omp critical
        printf("\n");
    }

    for (unsigned int i = 0; i < factorsLength; i++) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
        CHECK_CUDA_ERROR(cudaEventDestroy(startk[i]));
        CHECK_CUDA_ERROR(cudaEventDestroy(endk[i]));
        CHECK_CUDA_ERROR(cudaFree(d_matrix[i]));
        CHECK_CUDA_ERROR(cudaFreeHost(h_results[i]));
    }
    CHECK_CUDA_ERROR(cudaFreeHost(h_matrix));

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
