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

    //without omp
    gcc oclHeterogDevicesProc.c ../C_common/wtime.c ../C_common/device_info.c -O3 -lm  -D DEVICE= -framework OpenCL -I ../C_common -o oclHeterogDevicesProc

https://github.com/HandsOnOpenCL/Exercises-Solutions
https://github.com/smistad/OpenCL-Getting-Started/
https://github.com/essentialsofparallelcomputing/Chapter12/blob/master/OpenCL
*/


#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <time.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                \n" \
"   __global float* b,                                                \n" \
"   __global float* c,                                                \n" \
"   const unsigned int count)                                         \n" \
"{                                                                    \n" \
"   int i = get_global_id(0);                                         \n" \
"   if(i < count)                                                     \n" \
"       c[i] = a[i] + b[i];                                           \n" \
"}                                                                    \n";

int validate_results(const float* h_a, const float* h_b, const float* h_c, int start_idx, int length, const char* device_name) {
    int correct = 0;

    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int i = 0; i < length; i++) {
        float expected = h_a[start_idx + i] + h_b[start_idx + i];
        if (fabs(h_c[i] - expected) < TOL) {
            #pragma omp critical
            {
                printf(" %s %f h_a[%d] %f h_b[%d] %f h_c[%d] %f \n", 
                       device_name, expected, start_idx + i, h_a[start_idx + i], 
                       start_idx + i, h_b[start_idx + i], i, h_c[i]);
            }
            correct++;
        }
    }

    return correct;
}

int main(int argc, char** argv) {
    int err;
    int count = LENGTH;

    float* h_a = (float*) calloc(LENGTH, sizeof(float));
    float* h_b = (float*) calloc(LENGTH, sizeof(float));
    float* h_c_cpu = (float*) calloc(LENGTH / 2, sizeof(float));
    float* h_c_gpu = (float*) calloc(LENGTH / 2, sizeof(float));

    //Fill vectors a and b with random float values
    for (int i = 0; i < LENGTH; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    //Setup platform and devices
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");

    if (numPlatforms == 0) {
        printf("No platforms found. Exiting.\n");
        return EXIT_FAILURE;
    }

    cl_platform_id platforms[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    checkError(err, "Getting platforms");

    cl_command_queue queue_cpu = NULL, queue_gpu = NULL;

    //Find CPU and GPU devices
    cl_device_id device_cpu = NULL, device_gpu = NULL;
    for (int i = 0; i < numPlatforms; i++) {
        if (!device_cpu) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device_cpu, NULL);
            err = output_device_info(device_cpu);
        }
        if (!device_gpu) {
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_gpu, NULL);
            err = output_device_info(device_gpu);
        }
        if (device_cpu && device_gpu) break;
    }

    if (!device_cpu || !device_gpu) {
        printf("Failed to find both CPU and GPU devices. Exiting.\n");
        return EXIT_FAILURE;
    }

    //Create contexts and command queues
    cl_context context_cpu = clCreateContext(NULL, 1, &device_cpu, NULL, NULL, &err);
    checkError(err, "Creating CPU context");

    cl_context context_gpu = clCreateContext(NULL, 1, &device_gpu, NULL, NULL, &err);
    checkError(err, "Creating GPU context");


    #ifdef __APPLE__ 
        queue_cpu = clCreateCommandQueue(context_cpu, device_cpu, 0, &err);
        checkError(err, "Creating CPU command queue");
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create command queue (deprecated). Error code: %d\n", err);
            return -1;
        }
    #else  //Other platforms
        cl_command_queue_properties cpu_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

        queue_cpu = clCreateCommandQueueWithProperties(context_cpu, device_cpu, cpu_properties, &err);
        checkError(err, "Creating CPU command queue");
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create command queue with properties. Error code: %d\n", err);
            return -1;
        }
    #endif

    #ifdef __APPLE__ 
        queue_gpu = clCreateCommandQueue(context_gpu, device_gpu, 0, &err);
        checkError(err, "Creating GPU command queue");
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create command queue (deprecated). Error code: %d\n", err);
            return -1;
        }
    #else  //Other platforms
        cl_command_queue_properties gpu_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

        queue_gpu = clCreateCommandQueueWithProperties(context_gpu, device_gpu, gpu_properties, &err);
        checkError(err, "Creating GPU command queue");
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create command queue with properties. Error code: %d\n", err);
            return -1;
        }
    #endif


    //Create programs and kernels
    cl_program program_cpu = clCreateProgramWithSource(context_cpu, 1, &KernelSource, NULL, &err);
    checkError(err, "Creating CPU program");

    cl_program program_gpu = clCreateProgramWithSource(context_gpu, 1, &KernelSource, NULL, &err);
    checkError(err, "Creating GPU program");

    err = clBuildProgram(program_cpu, 1, &device_cpu, NULL, NULL, NULL);
    checkError(err, "Building CPU program");

    err = clBuildProgram(program_gpu, 1, &device_gpu, NULL, NULL, NULL);
    checkError(err, "Building GPU program");

    cl_kernel kernel_cpu = clCreateKernel(program_cpu, "vadd", &err);
    checkError(err, "Creating CPU kernel");

    cl_kernel kernel_gpu = clCreateKernel(program_gpu, "vadd", &err);
    checkError(err, "Creating GPU kernel");

    // Allocate half workload to CPU and GPU
    int half = LENGTH / 2;

    cl_mem d_a_cpu = clCreateBuffer(context_cpu, CL_MEM_READ_ONLY, sizeof(float) * half, NULL, &err);
    cl_mem d_b_cpu = clCreateBuffer(context_cpu, CL_MEM_READ_ONLY, sizeof(float) * half, NULL, &err);
    cl_mem d_c_cpu = clCreateBuffer(context_cpu, CL_MEM_WRITE_ONLY, sizeof(float) * half, NULL, &err);

    cl_mem d_a_gpu = clCreateBuffer(context_gpu, CL_MEM_READ_ONLY, sizeof(float) * half, NULL, &err);
    cl_mem d_b_gpu = clCreateBuffer(context_gpu, CL_MEM_READ_ONLY, sizeof(float) * half, NULL, &err);
    cl_mem d_c_gpu = clCreateBuffer(context_gpu, CL_MEM_WRITE_ONLY, sizeof(float) * half, NULL, &err);

    //Write data to buffers
    clEnqueueWriteBuffer(queue_cpu, d_a_cpu, CL_TRUE, 0, sizeof(float) * half, h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue_cpu, d_b_cpu, CL_TRUE, 0, sizeof(float) * half, h_b, 0, NULL, NULL);

    clEnqueueWriteBuffer(queue_gpu, d_a_gpu, CL_TRUE, 0, sizeof(float) * half, h_a + half, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue_gpu, d_b_gpu, CL_TRUE, 0, sizeof(float) * half, h_b + half, 0, NULL, NULL);

    //Set kernel arguments
    clSetKernelArg(kernel_cpu, 0, sizeof(cl_mem), &d_a_cpu);
    clSetKernelArg(kernel_cpu, 1, sizeof(cl_mem), &d_b_cpu);
    clSetKernelArg(kernel_cpu, 2, sizeof(cl_mem), &d_c_cpu);
    clSetKernelArg(kernel_cpu, 3, sizeof(unsigned int), &half);

    clSetKernelArg(kernel_gpu, 0, sizeof(cl_mem), &d_a_gpu);
    clSetKernelArg(kernel_gpu, 1, sizeof(cl_mem), &d_b_gpu);
    clSetKernelArg(kernel_gpu, 2, sizeof(cl_mem), &d_c_gpu);
    clSetKernelArg(kernel_gpu, 3, sizeof(unsigned int), &half);


    size_t global = half;
    
    double cpu_time = 0.0, gpu_time = 0.0;

    //Execute kernels
    clock_t start_cpu = clock();
    clEnqueueNDRangeKernel(queue_cpu, kernel_cpu, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(queue_cpu);
    clock_t end_cpu = clock();
    cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    clock_t start_gpu = clock();
    clEnqueueNDRangeKernel(queue_gpu, kernel_gpu, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(queue_gpu);
    clock_t end_gpu = clock();
    gpu_time = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    //Read results
    clEnqueueReadBuffer(queue_cpu, d_c_cpu, CL_TRUE, 0, sizeof(float) * half, h_c_cpu, 0, NULL, NULL);
    clEnqueueReadBuffer(queue_gpu, d_c_gpu, CL_TRUE, 0, sizeof(float) * half, h_c_gpu, 0, NULL, NULL);

    //Validate CPU results
    int correct_cpu = validate_results(h_a, h_b, h_c_cpu, 0, half, "CPU");
    printf("CPU Validation: %d out of %d correct.\n", correct_cpu, half);

    //Validate GPU results
    int correct_gpu = validate_results(h_a, h_b, h_c_gpu, half, half, "GPU");
    printf("GPU Validation: %d out of %d correct.\n", correct_gpu, half);

    //Print execution times
    printf("CPU Kernel Execution Time: %.8f seconds\n", cpu_time);
    printf("GPU Kernel Execution Time: %.8f seconds\n", gpu_time);

    //Cleanup resources
    clReleaseMemObject(d_a_cpu);
    clReleaseMemObject(d_b_cpu);
    clReleaseMemObject(d_c_cpu);
    clReleaseMemObject(d_a_gpu);
    clReleaseMemObject(d_b_gpu);
    clReleaseMemObject(d_c_gpu);

    clReleaseKernel(kernel_cpu);
    clReleaseKernel(kernel_gpu);
    clReleaseProgram(program_cpu);
    clReleaseProgram(program_gpu);

    clReleaseCommandQueue(queue_cpu);
    clReleaseCommandQueue(queue_gpu);
    clReleaseContext(context_cpu);
    clReleaseContext(context_gpu);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}