#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

extern double wtime();
extern int output_device_info(cl_device_id );

#define MAX_SOURCE_SIZE (0x100000)

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

/*
Student: Tiago de Souza Oliveira
    email: ti.olive@gmail.com
Universidade de Santiago de Compotela
Master on High Performance Computing
Professor: Juan Carlos Pichel Campos
14/jan/2024

To compile and run:
    compute --gpu
    module load cesga/2020 pocl/1.6-CUDA-system
    make
    
    ./oclHeterogDevicesProc

This code aims to demonstrate the use of OpenCL on heterogeneous devices.
The code is based on the vector_add example from the HandsOnOpenCL repository with some additional ideas and utils functions from
the OpenCL-Getting-Started repository.

To do so, the code is structured as following the steps in a way the global workload is split into one part for the CPU and one part for the GPU:
    1. Create two vectors A and B containing LIST_SIZE elements
    2. Filling the vectors with data
    3. Load the kernel source code into the array source_str

    4. Get all available platforms information
    5. Get device information for each platform
    6. Create oCL Context for CPU
    7. Create command queue for CPU
    8. Create memory buffers on the CPU for each vector 
    9. Copy the vectors A and B to their respective memory buffers on CPU (the first half of the workload)
    10. Create a program from the kernel source
    11. Build the program
    12. Create the OpenCL kernel
    13. Set the arguments of the kernel
    14. Enqueue kernel execution for CPU (the first half of the workload)

    15. Create oCL Context for GPU
    16. Create command queue for GPU
    17. Create memory buffers on the GPU for each vector 
    18. Copy the vectors A and B to their respective memory buffers on GPU (the second half of the workload)
    19. Create a program from the kernel source
    20. Build the program
    21. Create the OpenCL kernel
    22. Set the arguments of the kernel
    23. Enqueue kernel execution for GPU (the second half of the workload)

    24. Read the memory buffer C on the device to the local variable C
    25. Display the result to the screen

    26. Cleanup

https://github.com/HandsOnOpenCL/Exercises-Solutions
https://github.com/smistad/OpenCL-Getting-Started/
*/

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

int main(void) {
    int          err;               // error code returned from OpenCL calls

    float*       h_a = (float*) calloc(LENGTH, sizeof(float));       // a vector
    float*       h_b = (float*) calloc(LENGTH, sizeof(float));       // b vector
    float*       h_c = (float*) calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device

    unsigned int correct;           // number of correct results

    size_t global;                  // global domain size


    cl_uint numPlatforms;
    cl_platform_id *platforms;
    cl_int err;

    // Step 1: Get all platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        fprintf(stderr, "Failed to find any OpenCL platforms. Error code: %d\n", err);
        return -1;
    }

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform IDs. Error code: %d\n", err);
        free(platforms);
        return -1;
    }

    printf("Number of platforms: %d\n", numPlatforms);

    // Step 2: Iterate through platforms and query devices
    for (cl_uint i = 0; i < numPlatforms; i++) {
        printf("\nPlatform %d:\n", i + 1);

        // Query CPU devices
        cl_uint numCPUDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &numCPUDevices);
        if (err == CL_SUCCESS && numCPUDevices > 0) {
            printf("  CPU devices available: %d\n", numCPUDevices);
            cl_device_id *cpuDevices = (cl_device_id *)malloc(sizeof(cl_device_id) * numCPUDevices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, numCPUDevices, cpuDevices, NULL);
            
            for (cl_uint j = 0; j < numCPUDevices; j++) {
                char deviceName[128];
                clGetDeviceInfo(cpuDevices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
                printf("    CPU Device %d: %s\n", j + 1, deviceName);
            }
            free(cpuDevices);
        } else {
            printf("  No CPU devices found on this platform.\n");
        }

        // Query GPU devices
        cl_uint numGPUDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numGPUDevices);
        if (err == CL_SUCCESS && numGPUDevices > 0) {
            printf("  GPU devices available: %d\n", numGPUDevices);
            cl_device_id *gpuDevices = (cl_device_id *)malloc(sizeof(cl_device_id) * numGPUDevices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numGPUDevices, gpuDevices, NULL);
            
            for (cl_uint j = 0; j < numGPUDevices; j++) {
                char deviceName[128];
                clGetDeviceInfo(gpuDevices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
                printf("    GPU Device %d: %s\n", j + 1, deviceName);
            }
            free(gpuDevices);
        } else {
            printf("  No GPU devices found on this platform.\n");
        }
    }

    // Step 3: Free platform resources
    free(platforms);

    return 0;
}

