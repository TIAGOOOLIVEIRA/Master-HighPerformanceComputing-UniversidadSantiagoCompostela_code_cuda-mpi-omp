#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include <sys/resource.h>

    //https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
    //https://www.nvidia.com/en-us/on-demand/session/gtcspring23-S51897/?ncid=em-even-124008-vt33
    //https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
    //https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    //https://courses.nvidia.com/courses/course-v1:DLI+S-AC-01+V1/course/
    //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
    //https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
    //https://developer.nvidia.com/blog/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/
    //https://docs.nvidia.com/cuda/profiler-users-guide/index.html#remote-profiling
    //https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation__staged-concurrent-copy-and-execute


typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


/*
TODO: topic to explore: Profiling GPU Applications with Nsight Systems
To compile, profiling and running:
    compute --gpu
    module load cuda/11.0
    
    nvcc -pg -G -c -use_fast_math multiplematrixscaling.cu -o multiplematrixscaling
    nvprof --export-profile multiplematrixscaling_timeline.prof -f --analysis-metrics ./multiplematrixscaling
    sudo nvvp multiplematrixscaling_timeline.prof

    
    ./multiplematrixscaling
    
    sbatch job_multmascaling.sh
    watch -n 1 squeue -u curso370
*/

__global__ void matrix_scaling_factor_kernel_cuda(float *data, unsigned int N, float factor, unsigned int repeat);


//To initialize the matrix with random values
void init_matrix(Matrix *m, int width, int height){
    m->width = width;
    m->height = height;
        
    for(unsigned int i = 0; i < height; i++){
        for(unsigned int j = 0; j < width; j++){
            unsigned int idx2d = i * width + j; //1D flat index
            m->elements[idx2d] = i + j + 1;
        }
    } 
}

int main(int argc, char *argv[]) {
    unsigned int N = 1 << 3;
    unsigned int nn = N * N;
    unsigned int repeat = 2;
    float factors[] = {0.1, 0.2};

    unsigned int factorsLength = sizeof(factors) / sizeof(factors[0]);
    unsigned int sizeBytes = nn * sizeof(float); 

    Matrix in, inGPU;

    // Allocate pinned memory on the host
    cudaMallocHost(&in.elements, sizeBytes);
    init_matrix(&in, N, N);
    printf("\nCreated and populated input Matrix in.height:%u x in.width:%u", in.height, in.width);

    inGPU.width = in.width; 
    inGPU.height = in.height;
    cudaMallocManaged(&inGPU.elements, sizeBytes);

    // Copy matrix to GPU
    cudaMemcpy(inGPU.elements, in.elements, sizeBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (nn + threadsPerBlock - 1) / threadsPerBlock;
    printf("\nthreadsPerBlock: %u, blocksPerGrid: %u", threadsPerBlock, blocksPerGrid);

    cudaStream_t streams[factorsLength];
    cudaEvent_t startk[factorsLength], endk[factorsLength];

    // Create streams and events
    for (unsigned int i = 0; i < factorsLength; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&startk[i]);
        cudaEventCreate(&endk[i]);
    }

    // Launch kernels concurrently for each factor
    for (unsigned int i = 0; i < factorsLength; i++) {
        printf("\nLaunching kernel for factor: %f, stream ID:%u", factors[i], i);
        cudaEventRecord(startk[i], streams[i]);
        matrix_scaling_factor_kernel_cuda<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
            inGPU.elements, nn, factors[i], repeat
        );
        cudaEventRecord(endk[i], streams[i]);
    }

    // Wait for all streams to finish
    for (unsigned int i = 0; i < factorsLength; i++) {
        cudaStreamSynchronize(streams[i]);
        float millsevent = 0;
        cudaEventElapsedTime(&millsevent, startk[i], endk[i]);
        printf("\nKernel for factor: %f; Elapsed time: %f ms", factors[i], millsevent);
    }

    // Copy result back to host
    cudaMemcpy(in.elements, inGPU.elements, sizeBytes, cudaMemcpyDeviceToHost);

    // Print results
    printf("\n\nUpdated Matrix:");
    for (unsigned int i = 0; i < in.height; i++) {
        printf("\n");
        for (unsigned int j = 0; j < in.width; j++) {
            unsigned int idx2d = i * in.width + j; // 1D flat index
            printf("%f,", in.elements[idx2d]);
        }
    }

    // Clean up
    for (unsigned int i = 0; i < factorsLength; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(startk[i]);
        cudaEventDestroy(endk[i]);
    }

    cudaFreeHost(in.elements);
    cudaFree(inGPU.elements);

    return 0;
}


// CUDA kernel
__global__ void matrix_scaling_factor_kernel_cuda(float *data, unsigned int N, float factor, unsigned int repeat) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (col < N) {
        float val = data[col];
        for (unsigned int i = 0; i < repeat; i++) {
            val *= factor;
        }
        data[col] = val;  // Write the result back to the same array
    }
}


