#include "helper.cuh"
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_pipeline_primitives.h>
#include <cuda/pipeline>
#include <iostream>


namespace cg = cooperative_groups;

constexpr int ROWS = 20000;
constexpr int COLS = 20000;
constexpr int THREADS = 256;


__global__ void copy_pipeline(const float* __restrict__ input, float* __restrict__ output, int cols) {
    __shared__ float staging[2][THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    bool ping = 0;

    for (int j = tid; j < cols; j += blockDim.x) {
        cuda::memcpy_async(&staging[ping][tid], &input[row * cols + j],
                           cuda::aligned_size_t<8>(sizeof(float)), pipe);

        pipe.producer_commit();
        __pipeline_wait_prior(0);
        //pipe.consumer_release();

        //Do nothing with staging
        ping = !ping;
    }

    if (tid == 0) {
        output[row] = staging[ping][0];
    }
}

__global__ void copy_manual(const float* __restrict__ input, float* __restrict__ output, int cols) {
    __shared__ float staging[THREADS];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int j = tid; j < cols; j += blockDim.x) {
        staging[tid] = input[row * cols + j];
        __syncthreads();
    }

    if (tid == 0) {
        output[row] = staging[0];
    }
}

void run_kernel(const char* label, void (*kernel)(const float*, float*, int)) {
    size_t bytes = ROWS * COLS * sizeof(float);
    float* h_input = new float[ROWS * COLS];
    float* h_output = new float[ROWS];

    for (int i = 0; i < ROWS * COLS; i++)
        h_input[i] = static_cast<float>(i % 100);

    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, ROWS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    dim3 block(THREADS);
    dim3 grid(ROWS);

    std::cout << "\n--- " << label << " ---" << std::endl;

    Timer timer;
    timer.start();

    kernel<<<grid, block>>>(d_input, d_output, COLS);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    timer.stop(std::string("Execution time for ") + label);

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, ROWS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Sample output [0]: " << h_output[0] << std::endl;



    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    delete[] h_input;
    delete[] h_output;
}

int main() {
    run_kernel("Global-to-Shared Copy with cuda::pipeline", copy_pipeline);
    run_kernel("Global-to-Shared Copy with manual __syncthreads", copy_manual);
    return 0;
}