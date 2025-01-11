/*
Student: Tiago de Souza Oliveira
    email: ti.olive@gmail.com
Universidade de Santiago de Compotela
Master on High Performance Computing
Professor: Juan Carlos Pichel Campos
14/jan/2025

To compile and run:
    compute --gpu
    module load cesga/2020 pocl/1.6-CUDA-system
    make
    
    ./oclHeterogDevicesProc



https://github.com/HandsOnOpenCL/Exercises-Solutions
https://github.com/smistad/OpenCL-Getting-Started/
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

// Kernel source
const char *KernelSource = "__kernel void vadd(                        \n"
"   __global float* a,                                                \n"
"   __global float* b,                                                \n"
"   __global float* c,                                                \n"
"   const unsigned int count)                                         \n"
"{                                                                    \n"
"   int i = get_global_id(0);                                         \n"
"   if (i < count)                                                    \n"
"       c[i] = a[i] + b[i];                                           \n"
"}                                                                    \n";

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1 << 20)

// Utility to get current timestamp
void log_timestamp(const char *message) {
    time_t now;
    time(&now);
    printf("[%s] %s\n", ctime(&now), message);
}


void validation_loop(float *h_a1, float *h_b1, float *h_c1, float *h_a2, float *h_b2, float *h_c2, int count) {
    int correct = 0;  // Shared variable for correct count

    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int i = 0; i < count; i++) {
        float tmp1 = h_a1[i] + h_b1[i];  // assign element i of a+b to tmp1
        tmp1 -= h_c1[i];                 // compute deviation of expected and output result

        float tmp2 = h_a2[i] + h_b2[i];  // assign element i of a+b to tmp2
        tmp2 -= h_c2[i];                 // compute deviation of expected and output result

        if (tmp1 * tmp1 < TOL * TOL || tmp2 * tmp2 < TOL * TOL) {
            correct++;  // Increment correct count if within tolerance
        } else {
            #pragma omp critical
            {
                printf(" tmp1 %f h_a1 %f h_b1 %f h_c1 %f \n", tmp1, h_a1[i], h_b1[i], h_c1[i]);
                printf(" tmp2 %f h_a2 %f h_b2 %f h_c2 %f \n", tmp2, h_a2[i], h_b2[i], h_c2[i]);
            }
        }
    }

    // summarise results
    printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

    if (correct == count) {
        printf("Validation successful. Results match.\n");
    } else {
        printf("Validation failed. Errors: %d\n", correct);
    }
}


int main() {
    int err;
    float *h_a1 = (float *)calloc(LENGTH, sizeof(float));
    float *h_b1 = (float *)calloc(LENGTH, sizeof(float));
    float *h_c1 = (float *)calloc(LENGTH, sizeof(float));

    float *h_a2 = (float *)calloc(LENGTH, sizeof(float));
    float *h_b2 = (float *)calloc(LENGTH, sizeof(float));
    float *h_c2 = (float *)calloc(LENGTH, sizeof(float));

    int count = LENGTH;

    // Initialize data
    log_timestamp("Initializing input data...");
    #pragma omp parallel for
    for (int i = 0; i < LENGTH; i++) {
        h_a1[i] = rand() / (float)RAND_MAX;
        h_b1[i] = rand() / (float)RAND_MAX;
        h_a2[i] = rand() / (float)RAND_MAX;
        h_b2[i] = rand() / (float)RAND_MAX;
    }

    cl_uint numPlatforms;
    cl_platform_id *platforms;

    // Get platforms
    log_timestamp("Querying platforms...");
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        fprintf(stderr, "Failed to find any OpenCL platforms. Error code: %d\n", err);
        return -1;
    }

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    printf("Number of platforms: %d\n", numPlatforms);

    // Iterate through platforms and query devices
    cl_device_id device_cpu = NULL, device_gpu = NULL;
    cl_context context_cpu = NULL, context_gpu = NULL;
    cl_command_queue queue_cpu = NULL, queue_gpu = NULL;

    for (cl_uint i = 0; i < numPlatforms; i++) {
        printf("\nPlatform %d:\n", i + 1);

        // Query CPU devices
        cl_uint numCPUDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device_cpu, &numCPUDevices);
        if (err == CL_SUCCESS && numCPUDevices > 0) {
            printf("  CPU device found.\n");
            context_cpu = clCreateContext(NULL, 1, &device_cpu, NULL, NULL, &err);

            cl_command_queue_properties cpu_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

            queue_cpu = clCreateCommandQueueWithProperties(context_cpu, device_cpu, cpu_properties, &err);

        } else {
            printf("  No CPU devices found.\n");
        }

        // Query GPU devices
        cl_uint numGPUDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_gpu, &numGPUDevices);
        if (err == CL_SUCCESS && numGPUDevices > 0) {
            printf("  GPU device found.\n");
            context_gpu = clCreateContext(NULL, 1, &device_gpu, NULL, NULL, &err);

            cl_command_queue_properties gpu_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
            queue_gpu = clCreateCommandQueueWithProperties(context_gpu, device_gpu, gpu_properties, &err);
        } else {
            printf("  No GPU devices found.\n");
        }
    }

    free(platforms);

    // Ensure at least one device is available
    if (!device_cpu && !device_gpu) {
        fprintf(stderr, "No CPU or GPU devices found. Exiting.\n");
        return -1;
    }

    size_t global_size = LENGTH;
    size_t local_size = 256;

    cl_mem d_a1, d_b1, d_c1;
    cl_mem d_a2, d_b2, d_c2;

    // Create buffers for CPU
    if (context_cpu) {
        d_a1 = clCreateBuffer(context_cpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, h_a1, &err);
        d_b1 = clCreateBuffer(context_cpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, h_b1, &err);
        d_c1 = clCreateBuffer(context_cpu, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
    }

    // Create buffers for GPU
    if (context_gpu) {
        d_a2 = clCreateBuffer(context_gpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, h_a2, &err);
        d_b2 = clCreateBuffer(context_gpu, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, h_b2, &err);
        d_c2 = clCreateBuffer(context_gpu, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
    }

    // Create programs and kernels
    cl_program program_cpu = NULL, program_gpu = NULL;
    cl_kernel kernel_cpu = NULL, kernel_gpu = NULL;

    if (context_cpu) {
        program_cpu = clCreateProgramWithSource(context_cpu, 1, &KernelSource, NULL, &err);
        clBuildProgram(program_cpu, 1, &device_cpu, NULL, NULL, NULL);
        kernel_cpu = clCreateKernel(program_cpu, "vadd", &err);

        clSetKernelArg(kernel_cpu, 0, sizeof(cl_mem), &d_a1);
        clSetKernelArg(kernel_cpu, 1, sizeof(cl_mem), &d_b1);
        clSetKernelArg(kernel_cpu, 2, sizeof(cl_mem), &d_c1);
        clSetKernelArg(kernel_cpu, 3, sizeof(unsigned int), &count);
    }

    if (context_gpu) {
        program_gpu = clCreateProgramWithSource(context_gpu, 1, &KernelSource, NULL, &err);
        clBuildProgram(program_gpu, 1, &device_gpu, NULL, NULL, NULL);
        kernel_gpu = clCreateKernel(program_gpu, "vadd", &err);

        clSetKernelArg(kernel_gpu, 0, sizeof(cl_mem), &d_a2);
        clSetKernelArg(kernel_gpu, 1, sizeof(cl_mem), &d_b2);
        clSetKernelArg(kernel_gpu, 2, sizeof(cl_mem), &d_c2);
        clSetKernelArg(kernel_gpu, 3, sizeof(unsigned int), &count);
    }

    // Execute kernels and measure time
    double cpu_time = 0.0, gpu_time = 0.0;

    if (queue_cpu) {
        log_timestamp("Executing CPU kernel...");
        clock_t start_cpu = clock();
        clEnqueueNDRangeKernel(queue_cpu, kernel_cpu, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue_cpu);
        clock_t end_cpu = clock();
        cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    }

    if (queue_gpu) {
        log_timestamp("Executing GPU kernel...");
        clock_t start_gpu = clock();
        clEnqueueNDRangeKernel(queue_gpu, kernel_gpu, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue_gpu);
        clock_t end_gpu = clock();
        gpu_time = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    }

    // Read results
    if (queue_cpu) clEnqueueReadBuffer(queue_cpu, d_c1, CL_TRUE, 0, sizeof(float) * count, h_c1, 0, NULL, NULL);
    if (queue_gpu) clEnqueueReadBuffer(queue_gpu, d_c2, CL_TRUE, 0, sizeof(float) * count, h_c2, 0, NULL, NULL);

    // Print benchmarking results
    printf("CPU Kernel Execution Time: %.6f seconds\n", cpu_time);
    printf("GPU Kernel Execution Time: %.6f seconds\n", gpu_time);

    validation_loop(h_a1, h_b1, h_c1, h_a2, h_b2, h_c2, count);


    log_timestamp("Cleanup and exiting...");

    // Cleanup
    clReleaseMemObject(d_a1);
    clReleaseMemObject(d_b1);
    clReleaseMemObject(d_c1);
    clReleaseMemObject(d_a2);
    clReleaseMemObject(d_b2);
    clReleaseMemObject(d_c2);
    clReleaseKernel(kernel_cpu);
    clReleaseKernel(kernel_gpu);
    clReleaseProgram(program_cpu);
    clReleaseProgram(program_gpu);
    clReleaseCommandQueue(queue_cpu);
    clReleaseCommandQueue(queue_gpu);
    clReleaseContext(context_cpu);
    clReleaseContext(context_gpu);

    free(h_a1);
    free(h_b1);
    free(h_c1);
    free(h_a2);
    free(h_b2);
    free(h_c2);

    return 0;
}