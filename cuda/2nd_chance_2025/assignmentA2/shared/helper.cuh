#ifndef HELPER_CUH
#define HELPER_CUH

#include <iostream>
#include <chrono>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call)                                                \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (code %d)\n",                    \
                    cudaGetErrorString(err), err);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

//GPU Timer-CUDA events
class Timer {
    float time_ms;
    const int device;
    cudaEvent_t start_evt, stop_evt;
public:
    Timer(int gpu_device = 0) : device(gpu_device) {
        cudaSetDevice(device);
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }
    ~Timer() {
        cudaSetDevice(device);
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
    void start() {
        cudaSetDevice(device);
        cudaEventRecord(start_evt, 0);
    }
    void stop(const std::string &label) {
        cudaSetDevice(device);
        cudaEventRecord(stop_evt, 0);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time_ms, start_evt, stop_evt);
        std::cout << "GPU TIMING [" << label << "]: " << time_ms << " ms\n";
    }
    float milliseconds() const { return time_ms; }
};

//CPU Timer
class CpuTimer {
    std::chrono::high_resolution_clock::time_point t_start, t_stop;
    double time_ms;
public:
    void start() { t_start = std::chrono::high_resolution_clock::now(); }
    void stop(const std::string &label) {
        t_stop = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<double, std::milli>(t_stop - t_start).count();
        std::cout << "CPU TIMING [" << label << "]: " << time_ms << " ms\n";
    }
    double milliseconds() const { return time_ms; }
};

//device properties and theoretical occupancy for a given kernel
inline void print_device_info(const void *kernelFunc = nullptr,
                              int threadsPerBlock = 256,
                              int dynamicSMem = 0) {
    cudaDeviceProp prop;
    int dev;
    CHECK_CUDA_ERROR(cudaGetDevice(&dev));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev));

    printf("\n===|Begin| CUDA Device Info ===\n");
    printf("Device %d: %s\n", dev, prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f MB\n", prop.totalGlobalMem / (1024.0*1024.0));
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);

    if (kernelFunc) {
        int maxActiveBlocks = 0;
        CHECK_CUDA_ERROR(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocks,
                kernelFunc,
                threadsPerBlock,
                dynamicSMem
            )
        );
        float occupancy = (maxActiveBlocks * threadsPerBlock) /
                          float(prop.maxThreadsPerMultiProcessor) * 100.0f;
        printf("Theoretical Occupancy (%d tpb): %.1f%% (%d blocks/SM)\n",
               threadsPerBlock, occupancy, maxActiveBlocks);
    }
    printf("\n===|End| CUDA Device Info ===\n");

}

#endif // HELPER_CUH