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
To compile, profiling and running:
    compute --gpu
    nvcc -g -G -c multiplematrixscaling.cu -o multiplematrixscaling.o
    sudo nvpp ./multiplematrixscaling
    sudo nvprof --unified-memory-profiling off ./multiplematrixscaling
    nvcc -o multiplematrixscaling multiplematrixscaling.cu 
    
    ./multiplematrixscaling
    
    sbatch job_multmascaling.sh
    watch -n 1 squeue -u curso370
*/

__global__ void matrix_scaling_factor_kernel_cuda(const float *in, float *out, unsigned int N, float factor, unsigned int repeat);


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

int main(int argc, char *argv[])
{
    //x << y means x * (2^y) or x * pow(2, y)
    unsigned int N = 1 << 3;
    unsigned int nn = N * N;
    unsigned int repeat = 2;
    float factors[] = {0.1, 0.2};

    //for getting the lenght of the array
    //in case of iterating over the array for calling kernel passing each element factor
    unsigned int factorsLength = sizeof(factors) / sizeof(factors[0]);
    unsigned int sizeBytes = nn * sizeof(float); 


    Matrix in, inGPU;

    //Allocates page-locked memory on the host for the GPU input.
    cudaMallocHost(&in.elements, sizeBytes);



    init_matrix(&in, N, N);
    printf("\nCreated and populated input Matrix in.height:%u x in.width:%u", in.height, in.width);

    inGPU.width = in.width; inGPU.height = in.height;

    cudaMallocManaged(&inGPU.elements, sizeBytes);

    int threadsPerBlock = 256;
    int blocksPerGrid = (nn + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nthreadsPerBlock: %u, blocksPerGrid: %u", threadsPerBlock, blocksPerGrid);
    dim3 BLOCKS(blocksPerGrid);
    dim3 THREADS(threadsPerBlock);
    

    cudaMemcpy(inGPU.elements, in.elements, sizeBytes, cudaMemcpyHostToDevice);

    /*
    iterating over factors, taking into account streaming GPU memory for overlapping execution/data transferring CPU-GPU
    */
    cudaStream_t streams[factorsLength];
    float *resultsGPU[factorsLength];
    float *resultsToCPU[factorsLength];

    cudaEvent_t startm, stopm;
    cudaEventCreate(&startm);
    cudaEventCreate(&stopm);

    cudaEvent_t startk[factorsLength];
    cudaEvent_t endk[factorsLength];

    //create streams and allocate outputs for each stream
    for(unsigned int i = 0; i < factorsLength; i++){
        cudaStreamCreate(&streams[i]);
        cudaMallocManaged(&resultsGPU[i], sizeBytes);
        cudaMallocHost(&resultsToCPU[i], sizeBytes);
        
        cudaEventCreate(&startk[i]);
        cudaEventCreate(&endk[i]);
    }
    


    //launch kernels concurrently for each factor on streams
    for(unsigned int i = 0; i < factorsLength; i++){
        //kernel for factor 0.1/ stream1
        printf("\nLaunching kernel for factor: %f, stream ID:%u, CPU->GPU input", factors[i], i);
        float millsevent = 0;
        cudaEventRecord(startk[i], streams[i]);
        matrix_scaling_factor_kernel_cuda<<<BLOCKS, THREADS, 0, streams[i]>>>(inGPU.elements, resultsGPU[i], nn, factors[i], repeat);
        cudaEventRecord(endk[i], streams[i]);
        cudaEventSynchronize(endk[i]);
        cudaEventElapsedTime(&millsevent, startk[i], endk[i]);

        printf("\nLaunched kernel for factor: %f; Elapsed time: %f ms", factors[i], millsevent);
    }


    float millismem = 0;

    cudaEventRecord(startm);
    //copy from Stream[i] to CPU memory: resultsToCPU[i]
    for(unsigned int i = 0; i < factorsLength; i++){
        //to copy asynchronously from GPU to CPU memory and validate the result as sequential execution<->transferring between kernel
        cudaStreamSynchronize(streams[i]);
        printf("\nCopying from GPU to CPU memory, stream ID:%u, GPU->CPU output.", i);
        if(cudaMemcpyAsync(resultsToCPU[i], resultsGPU[i], sizeBytes, cudaMemcpyDeviceToHost) != cudaSuccess){
            printf("\nError copying memory cudaMemcpyDeviceToHost, index %u\n", i);

            for(unsigned int j = 0; j < factorsLength; j++){
                cudaStreamDestroy(streams[j]);
                cudaFree(resultsGPU[j]);
                cudaFreeHost(resultsToCPU[j]);
            }

            cudaFreeHost(in.elements);
            cudaFree(inGPU.elements);

            return 0;
        }
    }

    cudaEventRecord(stopm);

    cudaEventSynchronize(stopm);
    cudaEventElapsedTime(&millismem, startm, stopm);
    
    printf("\n\nElapsed time for copying GPU->CPU data for %u kernel(s) executions: %f ms", factorsLength, millismem);
    printf("\nEffective Bandwidth (GB/s): %fn", nn*4*3/millismem/1e6);

    //printf("\nFirst element of First factor result, [0][0]: %f, resultsGPU[0][0]: %f", resultsToCPU[0][0], resultsGPU[0][0]);

    
    //some validations
    printf("\n\nOriginal Matrix:");
    for(unsigned int i = 0; i < in.height; i++){
        printf("\n");
        for(unsigned int j = 0; j < in.width; j++){
        unsigned int idx2d = i * in.width + j; //1D flat index
        printf("%f,", in.elements[idx2d]);
        }
    } 

    printf("\n\nResults from GPU per stream/factor:");
    for(unsigned int strs = 0; strs < factorsLength; strs++){
        printf("\n\nFactor: %f", factors[strs]);
        for(unsigned int i = 0; i < N; i++){
            printf("\n");
            for(unsigned int j = 0; j < N; j++){
                unsigned int idx2d = i * N + j; //1D flat index
                printf("%f,", resultsToCPU[strs][idx2d]);
            }
        } 
    }
    

    for(unsigned int i = 0; i < factorsLength; i++){
        cudaStreamDestroy(streams[i]);
        cudaFree(resultsGPU[i]);
        cudaFreeHost(resultsToCPU[i]);
    }

    cudaFree(in.elements);
    cudaFree(inGPU.elements);

}

// CUDA kernel
__global__ void matrix_scaling_factor_kernel_cuda(const float *in, float *out, unsigned int N, float factor, unsigned int repeat){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    //printf("\nblockIdx.x: %u, blockDim.x: %u, threadIdx.x: %u, col: %u", blockIdx.x, blockDim.x, threadIdx.x, col);
    //boundary check
    if(col < N){
        float val = in[col];
        
        for(unsigned int i=0; i<repeat; i++){
            val *= factor;
        }
        
        out[col] = val;

        //printf("\nout[%u]: %f", col, out[col]);
    }
}

