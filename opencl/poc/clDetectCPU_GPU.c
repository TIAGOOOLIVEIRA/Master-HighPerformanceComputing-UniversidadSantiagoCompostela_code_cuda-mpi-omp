#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
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
