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

int main(void) {
    int i;
    const int LIST_SIZE = 1024;

    // Split the global size into two for CPU and GPU
    size_t cpuGlobalSize = LIST_SIZE / 2;
    size_t gpuGlobalSize = LIST_SIZE - cpuGlobalSize;

    // Offsets to divide the workload between CPU and GPU
    size_t cpuOffset = 0;
    size_t gpuOffset = cpuGlobalSize;

    //1. Create two vectors A and B containing LIST_SIZE elements    
    int halfSize = LIST_SIZE / 2;
    int *AfirstHalf = (int*)malloc(sizeof(int) * halfSize);
    int *AsecondHalf = (int*)malloc(sizeof(int) * halfSize);

    int *BfirstHalf = (int*)malloc(sizeof(int) * halfSize);
    int *BsecondHalf = (int*)malloc(sizeof(int) * halfSize);


    //2. Filling the vectors with data
    printf("\n\nFilling the vectors with data");
    for(i = 0; i < halfSize; i++) {
        AfirstHalf[i] = i;
        BfirstHalf[i] = LIST_SIZE - i;

        AsecondHalf[LIST_SIZE - i] = LIST_SIZE - i;
        BsecondHalf[LIST_SIZE - i] = i;
    }
    printf("\n  Filled the vectors with data");

    //3. Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    printf("\nOpening the kernel source code vector_add_kernel.cl");
    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    printf("\n  Opened the kernel source code vector_add_kernel.cl");

    //idea here is to split the workload (A + B) into two parts: 
    //  first half of A array and B for CPU processing and the second parts for GPU processing 
    //  so each device will process half of the workload.
    //
    //It is also part of the task run the same code on different platforms (MacOS, Ubuntu+GPU, FTIII)
    //  so the code should be able to detect the available devices and run the kernel on each of them.
    //In the end some stastics should be collected and printed out for the comparision of the performance

    cl_int ret;

    //4. Get all available platforms information
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    //CPU and GPU devices
    cl_device_id cpuDevice = NULL;
    cl_device_id gpuDevice = NULL;
    cl_context cpuContext, gpuContext;
    cl_command_queue cpuQueue, gpuQueue;
 

    //Iterate over platforms to fetch CPU and GPU devices
    for (cl_uint i = 0; i < numPlatforms; i++) {
        //5. Get device information for each platform
        //Query CL_DEVICE_TYPE_CPU
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &cpuDevice, NULL) == CL_SUCCESS) {
            printf("\n\nCPU device found on platform %d\n\n", i);

            ret = output_device_info(cpuDevice);
                checkError(ret, "Error when Printing device CPU output");


            //6. Create oCL Context for CPU
            cpuContext = clCreateContext(NULL, 1, &cpuDevice, NULL, NULL, &ret);
                checkError(ret, "Error when creating the context on CPU");
            
            //7. Create command queue for CPU
            cpuQueue = clCreateCommandQueue(cpuContext, cpuDevice, 0, &ret);
                checkError(ret, "Error when Command Queue the program on CPU");
            
            //8. Create memory buffers on the CPU for each vector 
            unsigned int cpuSize = sizeof(int) * cpuGlobalSize;
            cl_mem a_mem_obj = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY, 
                cpuSize, NULL, &ret);
            cl_mem b_mem_obj = clCreateBuffer(cpuContext, CL_MEM_READ_ONLY,
                cpuSize, NULL, &ret);
            cl_mem c_mem_obj = clCreateBuffer(cpuContext, CL_MEM_WRITE_ONLY, 
                cpuSize, NULL, &ret);

            //9. Copy the vectors A and B to their respective memory buffers on CPU (the first half of the workload)
            ret = clEnqueueWriteBuffer(cpuQueue, a_mem_obj, CL_TRUE, 0,
                cpuSize, AfirstHalf, 0, NULL, NULL);
            ret |= clEnqueueWriteBuffer(cpuQueue, b_mem_obj, CL_TRUE, 0, 
                cpuSize, BfirstHalf, 0, NULL, NULL);
                    checkError(ret, "Error when copying A and B vectors to the CPU mem buffers");

            //10. Create a program from the kernel source
            cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, 
                    (const char **)&source_str, (const size_t *)&source_size, &ret);
                checkError(ret, "Error when creating the program on CPU");

            //11. Build the program
            ret = clBuildProgram(cpuProgram, 1, &cpuDevice, NULL, NULL, NULL);
                checkError(ret, "Error when building the program on CPU");

            //12. Create the OpenCL kernel
            cl_kernel cpuKernel = clCreateKernel(cpuProgram, "vector_add", &ret);
                checkError(ret, "Error when creating the Kernel on CPU");

            //13. Set the arguments of the kernel
            ret = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
            ret |= clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
            ret |= clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
                checkError(ret, "Error when Setting kernel CPU arguments");

            size_t localSize = 64; // Process in groups of 64

            double rtime = wtime();

            //14. Enqueue kernel execution for CPU (the first half of the workload)
            //Execution of a portion of the workload on the CPU
            ret = clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 1, NULL, &cpuGlobalSize, &localSize, 0, NULL, NULL);
                checkError(ret, "Error when enqueueing kernel on CPU");
    
            // Wait for the commands to complete before stopping the timer
            ret = clFinish(cpuQueue);
                checkError(ret, "Waiting for CPU kernel to finish");

            rtime = wtime() - rtime;
            printf("\nThe CPU Add kernel ran in %lf seconds\n",rtime);

            // Read the memory buffer C on the device to the local variable C
            int *C = (int*)malloc(cpuSize);
            ret = clEnqueueReadBuffer(cpuQueue, c_mem_obj, CL_TRUE, 0, 
                    cpuSize, C, 0, NULL, NULL);

            // Display the result to the screen
            for(i = 0; i < cpuSize; i++)
                printf("%d + %d = %d\n", AfirstHalf[i], BfirstHalf[i], C[i]);

            //local cleanup
            ret = clReleaseProgram(cpuProgram);
            ret |= clReleaseMemObject(a_mem_obj);
            ret |= clReleaseMemObject(b_mem_obj);
            ret |= clReleaseMemObject(c_mem_obj);
                checkError(ret, "Error when cleaning up CPU Program and Mem buffer objects");

            free(C);
        }

        //Query CL_DEVICE_TYPE_GPU
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &gpuDevice, NULL) == CL_SUCCESS) {
            printf("\n\nGPU device found on platform %d\n\n", i);
            ret = output_device_info(gpuDevice);
                checkError(ret, "Error when Printing device GPU output");

            //6. Create oCL Context for GPU
            gpuContext = clCreateContext(NULL, 1, &gpuDevice, NULL, NULL, &ret);
                checkError(ret, "Error when creating the context on GPU");
            
            //7. Create command queue for GPU
            gpuQueue = clCreateCommandQueue(gpuContext, gpuDevice, 0, &ret);
                checkError(ret, "Error when Command Queue the context on GPU");

            //8. Create memory buffers on the GPU for each vector 
            unsigned int gpuSize = sizeof(int) * gpuGlobalSize;
            cl_mem a_mem_obj = clCreateBuffer(gpuQueue, CL_MEM_READ_ONLY, 
                gpuSize, NULL, &ret);
            cl_mem b_mem_obj = clCreateBuffer(gpuQueue, CL_MEM_READ_ONLY,
                gpuSize, NULL, &ret);
            cl_mem c_mem_obj = clCreateBuffer(gpuQueue, CL_MEM_WRITE_ONLY, 
                gpuSize, NULL, &ret);

            //9. Copy the vectors A and B to their respective memory buffers on GPU (the second half of the workload)
            ret = clEnqueueWriteBuffer(gpuQueue, a_mem_obj, CL_TRUE, 0,
                gpuSize, AsecondHalf, 0, NULL, NULL);
            ret |= clEnqueueWriteBuffer(gpuQueue, b_mem_obj, CL_TRUE, 0, 
                gpuSize, BsecondHalf, 0, NULL, NULL);
                    checkError(ret, "Error when copying A and B vectors to the GPU mem buffers");


            //10. Create a program from the kernel source
            cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, 
                    (const char **)&source_str, (const size_t *)&source_size, &ret);
                checkError(ret, "Error when creating the program on GPU");

            //11. Build the program
            ret = clBuildProgram(gpuProgram, 1, &gpuDevice, NULL, NULL, NULL);
                checkError(ret, "Error when building the program on GPU");

            //12. Create the OpenCL kernel
            cl_kernel gpuKernel = clCreateKernel(gpuProgram, "vector_add", &ret);
                checkError(ret, "Error when creating the kernel on GPU");

            //13. Set the arguments of the kernel
            ret = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
            ret |= clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
            ret |= clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
                checkError(ret, "Error when Setting up GPU kernel arguments");

            size_t localSize = 64; // Process in groups of 64

            double rtime = wtime();

            //14. Enqueue kernel execution for GPU (the second half of the workload)
            ret = clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 1, &gpuOffset, &gpuGlobalSize, &localSize, 0, NULL, NULL);
                checkError(ret, "Error when enqueueing kernel on GPU");
        
            // Wait for the commands to complete before stopping the timer
            ret = clFinish(gpuQueue);
                checkError(ret, "Waiting for GPU kernel to finish");

            rtime = wtime() - rtime;
            printf("\nThe GPU Add kernel ran in %lf seconds\n",rtime);

            // Read the memory buffer C on the device to the local variable C
            int *C = (int*)malloc(gpuSize);
            ret = clEnqueueReadBuffer(cpuQueue, c_mem_obj, CL_TRUE, 0, 
                    gpuSize, C, 0, NULL, NULL);

            // Display the result to the screen
            for(i = 0; i < gpuSize; i++)
                printf("%d + %d = %d\n", AsecondHalf[i], BsecondHalf[i], C[i]);

            //local cleanup
            ret = clReleaseProgram(gpuProgram);
            ret |= clReleaseMemObject(a_mem_obj);
            ret |= clReleaseMemObject(b_mem_obj);
            ret |= clReleaseMemObject(c_mem_obj);
                checkError(ret, "Error when cleaning up GPU Program and Mem buffer objects");

            free(C);
        }
    }

    if (cpuQueue){
        ret = clFlush(cpuQueue);
        ret |= clFinish(cpuQueue);
        ret |= clReleaseCommandQueue(cpuQueue);
            checkError(ret, "Error when cleaning up CPU Command Queue");
    } 
    
    if (gpuQueue){
        ret = clFlush(gpuQueue);
        ret |= clFinish(gpuQueue);
        ret |= clReleaseCommandQueue(gpuQueue);
            checkError(ret, "Error when cleaning up GPU Command Queue");
    } 

    if (cpuContext) clReleaseContext(cpuContext);
    if (gpuContext) clReleaseContext(gpuContext);
    
    free(platforms);

    free(AfirstHalf);
    free(BfirstHalf);
    free(AsecondHalf);
    free(BsecondHalf);
    
    return 0;
}

