/* 
   Programación de GPUs (General Purpose Computation on Graphics Processing Unit)
©
   Margarita Amor López
   Emilio J. Padrón González

    This program finds the closest pair of points in a set of points in a 2D plane.
    Input Parameters:
    #n: size of the vectors

###########################################################################################
Student: Tiago de Souza Oliveira
Assignment 3 - 2023
           Assignment 3. Multi-levels of parallelism on GPU
*/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>


//https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
//CMSC 451: Closest Pair of Points https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/closepoints.pdf
/*
To compile, run and profile:
    compute --gpu
    nvcc -g -G -c MultiLevelsParallelism_ClosestPairPoints.cu -o MultiLevelsParallelism_ClosestPairPoints.o
    sudo nvpp ./MultiLevelsParallelism_ClosestPairPoints
    sudo nvprof --unified-memory-profiling off ./MultiLevelsParallelism_ClosestPairPoints 
    nvcc -o MultiLevelsParallelism_ClosestPairPoints MultiLevelsParallelism_ClosestPairPoints.cu -use_fast_math
    
    ./MultiLevelsParallelism_ClosestPairPoints
    
    sbatch job_MultiLevelsParallelism_ClosestPairPoints.sh
    watch -n 1 squeue -u curso370

-use_fast_math
*/

//strcut to represent a point with x and y coordinates
struct Point
{
    int x, y;
};

const int N = 1024;    // Standard value for n Points in the array

__global__ void findClosestPointGPU(Point *points, unsigned int* indices, unsigned int n);

// Function to find distance between two points
__device__ float distGPU(Point p1, Point p2)
{
    //leveraging Intrinsics https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html
    // for the tradeoff between accuracy and performance
    return __fsqrt_rn(  (p1.x - p2.x) * (p1.x - p2.x) +
                    (p1.y - p2.y) * (p1.y - p2.y)
                );
}

float distCPU(Point p1, Point p2)
{
    //leveraging Intrinsics https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html
    // for the tradeoff between accuracy and performance
    return sqrt(  (p1.x - p2.x) * (p1.x - p2.x) +
                    (p1.y - p2.y) * (p1.y - p2.y)
                );
}

// Function to initialize matrix with random points
void init_matrix_random_points(Point *P, int n)
{
    for (int i = 0; i < n; ++i)
    {
        P[i].x = rand() % 10000;
        P[i].y = rand() % 10000;
    }
}


// A Brute Force method to return the smallest distance between two points
void closesPairCPU(Point P[], unsigned int* indices, unsigned int n)
{
    float min = FLT_MAX;
    for (int curPoint = 0; curPoint < n; ++curPoint)
        //curPoint+1 to ensure do not check distance between the same point
        for (int j = curPoint+1; j < n; ++j)
            if (distCPU(P[curPoint], P[j]) < min){
                min = distCPU(P[curPoint], P[j]);
                indices[curPoint] = j;
            }                
}

// Function to compare GPU and CPU results
void compareCPU_GPU(unsigned int *indicesGPU,unsigned int *indicesCPU, int n) {
    bool match = true;
    for (int i = 0; i < n; ++i) {
        if (indicesGPU[i] != indicesCPU[i]) {
            printf("\nGPU<->CPU result does not match at index %d: GPU:%d CPU:%d\n", i, indicesGPU[i], indicesCPU[i]);
            match = false;
        }
    }

    if (match) printf("\n\nGPU and CPU match.\n");
}


int main(int argc, char *argv[]) {
    if (argc == 0) {
        printf("Please inform the number of Points to create");
        return 1;
    }

    unsigned int n = (argc > 1)?atoi (argv[1]):N;
    unsigned int sizeOfPoints = n * sizeof(Point);
    unsigned int sizeOfIndex = n * sizeof(unsigned int);

    // Allocating host memory
    unsigned int *indexofClosestPoint = (unsigned int *)malloc(sizeOfIndex);
    unsigned int *indexofClosestPointCPU_Validation = (unsigned int *)malloc(sizeOfIndex);
    Point *points = (Point *)malloc(sizeOfPoints);

    init_matrix_random_points(points, n);

    //allocating device memory
    unsigned int *indexofClosestPointGPU;   
    Point *pointsGPU;
    cudaMalloc(&pointsGPU, sizeOfPoints);
    cudaMalloc(&indexofClosestPointGPU, sizeOfIndex);
    cudaMemset(indexofClosestPointGPU, 0, sizeOfIndex);

    cudaMemcpy(pointsGPU, points, sizeOfPoints, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("\nthreadsPerBlock: %u, blocksPerGrid: %u", threadsPerBlock, blocksPerGrid);
    dim3 BLOCKS(blocksPerGrid);
    dim3 THREADS(threadsPerBlock);

 
    findClosestPointGPU<<<BLOCKS, THREADS>>>(pointsGPU, indexofClosestPointGPU, n);
    
    cudaMemcpy(indexofClosestPoint, indexofClosestPointGPU, sizeOfIndex, cudaMemcpyDeviceToHost);

 
    //call CPU version
    closesPairCPU(points, indexofClosestPointCPU_Validation, n);

    //Compare GPU and CPU results
    compareCPU_GPU(indexofClosestPoint, indexofClosestPointCPU_Validation, n);

    //free memory GPU,CPU    
    cudaFree(pointsGPU);
    cudaFree(indexofClosestPointGPU);

    free(points);
    free(indexofClosestPoint);
    free(indexofClosestPointCPU_Validation);

    return 0;
}

//kernel responsible to find the closest point for each point in the array
__global__ void findClosestPointGPU(Point *points, unsigned int* indices, unsigned int n)
{
    extern __shared__ Point sharedpts[];

    unsigned int tid = threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;


    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridSize) {
        float minDist = FLT_MAX;
        int minIdx = -1;

        sharedpts[tid] = points[i];
        __syncthreads();

        //To find pairwise distances & closest pair
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            float distance = distGPU(sharedpts[tid], sharedpts[j]);

            if (distance < minDist) {
                minDist = distance;
                minIdx = j;
            }
            
        }
        
        // Use atomicMin to set the closest point in global memory of indices 
        atomicMin(&indices[i], minIdx);
        __syncthreads();
    }
    
}