#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

//compute --gpu
//export OMP_NUM_THREADS=4
//nvcc -o mfactorscaling_omp_efficientmem mfactorscaling_omp_efficientmem.cu -Xcompiler -fopenmp -lcudart
//time ./mfactorscaling_omp 128 100 100 "0.1,0.3"

// CUDA kernel for matrix and factor multiplication using shared memory for factors
__global__ void matrixFactorMultiply(const float *matrix, const float *factors, float *result, int n, int m, int numFactors) {
    // Allocate shared memory for the factors array
    __shared__ float shared_factors[10];

    if (threadIdx.x < numFactors) {
        shared_factors[threadIdx.x] = factors[threadIdx.x];
    }
    __syncthreads(); 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < n * m) {
        float accum = 0.0;

        // Accumulate the product of the matrix element and each factor
        for (int i = 0; i < numFactors; i++) {
            accum += matrix[idx] * shared_factors[i];
        }

        result[idx] = accum;
    }
}

// Function to parse comma-separated values into an array of floats
int parseFactors(const char *input, float *factors) {
    int count = 0;
    char *token;
    char *inputCopy = strdup(input);
    token = strtok(inputCopy, ",");
    while (token != NULL) {
        factors[count++] = atof(token);
        token = strtok(NULL, ",");
    }
    free(inputCopy);
    return count;
}

// CPU verification function with OpenMP parallelism to speed up the total time processing of the app
void verifyResult(const float *matrix, const float *factors, const float *gpuResult, int n, int m, int numFactors) {
    int size = n * m;
    float epsilon = 1e-5; // Tolerance for floating-point comparison
    int mismatch = 0;

    #pragma omp parallel for reduction(+:mismatch)
    for (int idx = 0; idx < size; idx++) {
        float accum = 0.0;
        for (int i = 0; i < numFactors; i++) {
            accum += matrix[idx] * factors[i];
        }

        // Check if the GPU result matches the CPU computed result within tolerance
        if (fabs(accum - gpuResult[idx]) > epsilon) {
            #pragma omp critical
            {
                printf("Verification failed at index %d: CPU = %f, GPU = %f\n", idx, accum, gpuResult[idx]);
                mismatch++;
            }
        }
    }

    if (mismatch == 0) {
        printf("GPU and CPU results match.\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Arguments: %s <threads_per_block> <rows (n)> <columns (m)> <factors (e.g., \"0.1,0.3\")>\n", argv[0]);
        return 1;
    }

    int threadsPerBlock = atoi(argv[1]);
    int n = atoi(argv[2]); // Matrix rows
    int m = atoi(argv[3]); // Matrix columns
    const char *factorsInput = argv[4];

    float h_factors[10];
    int numFactors = parseFactors(factorsInput, h_factors);

    // Host memory allocation
    float *h_matrix = (float *)malloc(n * m * sizeof(float));
    float *h_result = (float *)malloc(n * m * sizeof(float));
    float *h_gpuResult = (float *)malloc(n * m * sizeof(float));

    #pragma omp parallel for
    for (int i = 0; i < n * m; i++) {
        h_matrix[i] = 1.0f;
    }

    // Device memory allocation
    float *d_matrix, *d_factors, *d_result;
    cudaMalloc((void **)&d_matrix, n * m * sizeof(float));
    cudaMalloc((void **)&d_factors, numFactors * sizeof(float));
    cudaMalloc((void **)&d_result, n * m * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMemcpy(d_matrix, h_matrix, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, h_factors, numFactors * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block configuration
    dim3 blockSize(threadsPerBlock);
    dim3 gridSize((n * m + threadsPerBlock - 1) / threadsPerBlock);

    matrixFactorMultiply<<<gridSize, blockSize>>>(d_matrix, d_factors, d_result, n, m, numFactors);

    cudaMemcpy(h_gpuResult, d_result, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Mem logistic + GPU processing time: %.4f seconds\n", milliseconds / 1000.0f);

    verifyResult(h_matrix, h_factors, h_gpuResult, n, m, numFactors);

    //Clean up
    cudaFree(d_matrix);
    cudaFree(d_factors);
    cudaFree(d_result);
    free(h_matrix);
    free(h_gpuResult);
    free(h_result);
    
    cudaDeviceReset();
    return 0;
}