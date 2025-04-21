#ifndef HELPER_CUH
#define HELPER_CUH

#include <iostream>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call)                                                \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s (code %d)\n",                     \
                    cudaGetErrorString(err), err);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

//GPU Timer-CUDA events
class Timer {
    float time;
    const uint64_t gpu;
    cudaEvent_t start_evt, stop_evt;

public:
    Timer(uint64_t gpu = 0) : gpu(gpu) {
        cudaSetDevice(gpu);
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }

    ~Timer() {
        cudaSetDevice(gpu);
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }

    void start() {
        cudaSetDevice(gpu);
        cudaEventRecord(start_evt, 0);
    }

    void stop(const std::string& label) {
        cudaSetDevice(gpu);
        cudaEventRecord(stop_evt, 0);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time, start_evt, stop_evt);
        std::cout << "GPU TIMING [" << label << "]: " << time << " ms\n";
    }

    float milliseconds() const {
        return time;
    }
};

//CPU Timer
class CpuTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
    double duration_ms;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string& label) {
        stop_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop_time - start_time).count();
        std::cout << "CPU TIMING [" << label << "]: " << duration_ms << " ms\n";
    }

    double milliseconds() const {
        return duration_ms;
    }
};

#endif