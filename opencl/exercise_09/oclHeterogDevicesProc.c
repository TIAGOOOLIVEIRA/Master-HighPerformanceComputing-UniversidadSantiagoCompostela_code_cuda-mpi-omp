#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    int i;
    const int LIST_SIZE = 1024;

    //1. Create two vectors A and B containing LIST_SIZE elements
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);

    //2. Filling the vectors with data
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    //3. Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );


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
 

    // Split the global size into two for CPU and GPU
    size_t cpuGlobalSize = LIST_SIZE / 2;
    size_t gpuGlobalSize = LIST_SIZE - cpuGlobalSize;

    // Offsets to divide the workload between CPU and GPU
    size_t cpuOffset = 0;
    size_t gpuOffset = cpuGlobalSize;
    

    //Iterate over platforms to fetch CPU and GPU devices
    for (cl_uint i = 0; i < numPlatforms; i++) {
        //5. Get device information for each platform
        //Query CL_DEVICE_TYPE_CPU
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &cpuDevice, NULL) == CL_SUCCESS) {
            
            //6. Create oCL Context for CPU
            cpuContext = clCreateContext(NULL, 1, &cpuDevice, NULL, NULL, &ret);
            
            //7. Create command queue for CPU
            cpuQueue = clCreateCommandQueue(cpuContext, cpuDevice, 0, &ret);
            
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
                cpuSize, A, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(cpuQueue, b_mem_obj, CL_TRUE, 0, 
                cpuSize, B, 0, NULL, NULL);


            //10. Create a program from the kernel source
            cl_program cpuProgram = clCreateProgramWithSource(cpuContext, 1, 
                    (const char **)&source_str, (const size_t *)&source_size, &ret);

            //11. Build the program
            ret = clBuildProgram(cpuProgram, 1, &cpuDevice, NULL, NULL, NULL);
                checkError(ret, "Error when building the program on CPU");

            //12. Create the OpenCL kernel
            cl_kernel cpuKernel = clCreateKernel(cpuProgram, "vector_add", &ret);

            //13. Set the arguments of the kernel
            ret = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
            ret = clSetKernelArg(cpuKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
            ret = clSetKernelArg(cpuKernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

            size_t localSize = 64; // Process in groups of 64

            //14. Enqueue kernel execution for CPU (the first half of the workload)
            //Execution of a portion of the workload on the CPU
            ret = clEnqueueNDRangeKernel(cpuQueue, cpuKernel, 1, NULL, &cpuGlobalSize, &localSize, 0, NULL, NULL);
                checkError(ret, "Error when enqueueing kernel on CPU");

        
            // Read the memory buffer C on the device to the local variable C
            int *C = (int*)malloc(cpuSize);
            ret = clEnqueueReadBuffer(cpuQueue, c_mem_obj, CL_TRUE, 0, 
                    cpuSize, C, 0, NULL, NULL);

            // Display the result to the screen
            for(i = 0; i < cpuSize; i++)
                printf("%d + %d = %d\n", A[i], B[i], C[i]);

            ret = clReleaseProgram(cpuProgram);
            ret = clReleaseMemObject(a_mem_obj);
            ret = clReleaseMemObject(b_mem_obj);
            ret = clReleaseMemObject(c_mem_obj);

            free(C);
        }

        //Query CL_DEVICE_TYPE_GPU
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &gpuDevice, NULL) == CL_SUCCESS) {
            
            //6. Create oCL Context for GPU
            gpuContext = clCreateContext(NULL, 1, &gpuDevice, NULL, NULL, &ret);
            
            //7. Create command queue for GPU
            gpuQueue = clCreateCommandQueue(gpuContext, gpuDevice, 0, &ret);

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
                gpuSize, A + gpuGlobalSize, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(gpuQueue, b_mem_obj, CL_TRUE, 0, 
                gpuSize, B + gpuGlobalSize, 0, NULL, NULL);


            //10. Create a program from the kernel source
            cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1, 
                    (const char **)&source_str, (const size_t *)&source_size, &ret);

            //11. Build the program
            ret = clBuildProgram(gpuProgram, 1, &gpuDevice, NULL, NULL, NULL);
                checkError(ret, "Error when building the program on GPU");

            //12. Create the OpenCL kernel
            cl_kernel gpuKernel = clCreateKernel(gpuProgram, "vector_add", &ret);

            //13. Set the arguments of the kernel
            ret = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
            ret = clSetKernelArg(gpuKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
            ret = clSetKernelArg(gpuKernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

            size_t localSize = 64; // Process in groups of 64

            //14. Enqueue kernel execution for GPU (the second half of the workload)
            ret = clEnqueueNDRangeKernel(gpuQueue, gpuKernel, 1, &gpuOffset, &gpuGlobalSize, &localSize, 0, NULL, NULL);
                checkError(ret, "Error when enqueueing kernel on GPU");
        
            // Read the memory buffer C on the device to the local variable C
            int *C = (int*)malloc(gpuSize);
            ret = clEnqueueReadBuffer(cpuQueue, c_mem_obj, CL_TRUE, 0, 
                    gpuSize, C, 0, NULL, NULL);

            // Display the result to the screen
            for(i = 0; i < gpuSize; i++)
                printf("%d + %d = %d\n", A[i], B[i], C[i]);


            ret = clReleaseProgram(gpuProgram);
            ret = clReleaseMemObject(a_mem_obj);
            ret = clReleaseMemObject(b_mem_obj);
            ret = clReleaseMemObject(c_mem_obj);

            free(C);
        }
    }

    // Cleanup
    if (cpuQueue){
        clFlush(cpuQueue);
        clFinish(cpuQueue);
        clReleaseCommandQueue(cpuQueue);
    } 
    
    if (gpuQueue){
        clFlush(gpuQueue);
        clFinish(gpuQueue);
        clReleaseCommandQueue(gpuQueue);
    } 

    if (cpuContext) clReleaseContext(cpuContext);
    if (gpuContext) clReleaseContext(gpuContext);
    
    free(platforms);

    free(A);
    free(B);
    
    return 0;
}

