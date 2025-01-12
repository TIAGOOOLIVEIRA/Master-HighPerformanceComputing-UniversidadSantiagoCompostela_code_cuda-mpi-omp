/*
Student: Tiago de Souza Oliveira
    email: ti.olive@gmail.com
Universidade de Santiago de Compotela
Master on High Performance Computing
Professor: Juan Carlos Pichel Campos
12/jan/2025

To compile and run:
    compute --gpu
    module load cesga/2020 pocl/1.6-CUDA-system
    make
    
    ./pi_ocl


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

#define NUM_STEPS (1 << 20)

const char *kernelSource = "\n"
"__kernel void pi_wg(__global double* partial_sums, const unsigned int num_steps, const double step) {\n"
"    // Get global and local indices\n"
"    int global_id = get_global_id(0);\n"
"    int local_id = get_local_id(0);\n"
"    int group_id = get_group_id(0);\n"
"    int local_size = get_local_size(0);\n"
"\n"
"    //Allocate local memory for reduction\n"
"    __local double local_sum[256]; // Adjust size to match your max local workgroup size\n"
"\n"
"    //sum for the work item\n"
"    double x, sum = 0.0;\n"
"\n"
"    //Compute partial sum for the current work item\n"
"    for (int i = global_id; i < num_steps; i += get_global_size(0)) {\n"
"        x = (i + 0.5) * step;\n"
"        sum += 4.0 / (1.0 + x * x);\n"
"    }\n"
"\n"
"    //Store the partial sum in local memory\n"
"    local_sum[local_id] = sum;\n"
"\n"
"    //Sync\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"    //Tree-height reduction within the workgroup\n"
"    for (int stride = local_size / 2; stride > 0; stride /= 2) {\n"
"        if (local_id < stride) {\n"
"            local_sum[local_id] += local_sum[local_id + stride];\n"
"        }\n"
"        barrier(CLK_LOCAL_MEM_FENCE); // Synchronize after each step\n"
"    }\n"
"\n"
"    //Result of a workgroup to the global memory\n"
"    if (local_id == 0) {\n"
"        partial_sums[group_id] = local_sum[0];\n"
"    }\n"
"}\n";


int main(int argc, char** argv) {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem partial_sums_buffer;

    int num_steps = NUM_STEPS;

    double step = 1.0 / num_steps;
    size_t local_size = 256;
    size_t global_size = ((num_steps + local_size - 1) / local_size) * local_size;
    size_t num_workgroups = global_size / local_size;

    // Query platform and GPU device
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "Querying platform");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    //TODO
    //to make it work on Apple M1 Pro I had to change for iterating over the devices

    #ifdef __APPLE__ 
        char deviceName[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        printf("Device: %s\n", deviceName);
    #else
        err = output_device_info(device);
    #endif

    


    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to find a GPU device. Error code: %d\n", err);
        return EXIT_FAILURE;
    }

    // Create OpenCL context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    #ifdef __APPLE__ 
        queue = clCreateCommandQueue(context, device, 0, &err);
        checkError(err, "Creating command queue");
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create command queue (deprecated). Error code: %d\n", err);
            return -1;
        }
    #else
        cl_command_queue_properties cpu_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

        queue = clCreateCommandQueueWithProperties(context, device, cpu_properties, &err);
        checkError(err, "Creating command queue");
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to create command queue with properties. Error code: %d\n", err);
            return -1;
        }
    #endif

    // Create program and kernel
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    checkError(err, "Creating program");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Error building program:\n%s\n", build_log);
        return EXIT_FAILURE;
    }

    kernel = clCreateKernel(program, "pi_wg", &err);
    checkError(err, "Creating kernel");

    // Allocate buffer for partial sums
    partial_sums_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_workgroups, NULL, &err);
    checkError(err, "Creating partial sums buffer");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &partial_sums_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &num_steps);
    err |= clSetKernelArg(kernel, 2, sizeof(double), &step);
    checkError(err, "Setting kernel arguments");

    // Execute kernel
    clock_t start_cpu = clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    err = clFinish(queue);
    checkError(err, "Finishing queue");
    clock_t end_cpu = clock();
    double execution_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // Read back partial sums
    double *partial_sums = (double*) malloc(sizeof(double) * num_workgroups);
    err = clEnqueueReadBuffer(queue, partial_sums_buffer, CL_TRUE, 0, sizeof(double) * num_workgroups, partial_sums, 0, NULL, NULL);
    checkError(err, "Reading partial sums");

    // Perform final reduction on host
    double pi = 0.0;
    for (size_t i = 0; i < num_workgroups; i++) {
        pi += partial_sums[i];
    }
    pi *= step;

    printf("Calculated Pi: %.15f\n", pi);
    printf("Execution time: %.8f seconds\n", execution_time);

    free(partial_sums);
    clReleaseMemObject(partial_sums_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;


}

