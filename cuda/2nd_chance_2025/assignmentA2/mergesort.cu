#include "helper.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

// Merge function for device
__device__ void merge(int* __restrict__ data, int* __restrict__ temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        temp[k++] = (data[i] <= data[j]) ? data[i++] : data[j++];
    }
    while (i <= mid) temp[k++] = data[i++];
    while (j <= right) temp[k++] = data[j++];
    for (i = left; i <= right; i++) data[i] = temp[i];
}

// Iterative merge-sort kernel per-array
__global__ void sort_kernel(int* __restrict__ data, int size) {
    extern __shared__ int temp[];
    for (int width = 1; width < size; width <<= 1) {
        for (int idx = 0; idx < size; idx += (width << 1)) {
            int left = idx;
            int mid = min(idx + width - 1, size - 1);
            int right = min(idx + (width << 1) - 1, size - 1);
            if (mid < right) merge(data, temp, left, mid, right);
        }
        __syncthreads();
    }
}

// CPU-side merge and parallel merge sort for validation
void merge_cpu(int* __restrict__ arr, int l, int m, int r, int* __restrict__ temp) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        temp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; ++i) arr[i] = temp[i];
}

void merge_sort_cpu_parallel(int* __restrict__ arr, int l, int r, int* __restrict__ temp, int depth, bool parallel=true) {
    if (l < r) {
        int m = (l + r) / 2;
        if (depth <= 0) {
            merge_sort_cpu_parallel(arr, l, m, temp, depth - 1, parallel);
            merge_sort_cpu_parallel(arr, m + 1, r, temp, depth - 1, parallel);
        } else {
            #pragma omp parallel sections if(parallel)
            {
                #pragma omp section
                merge_sort_cpu_parallel(arr, l, m, temp, depth - 1, parallel);
                #pragma omp section
                merge_sort_cpu_parallel(arr, m + 1, r, temp, depth - 1, parallel);
            }
        }
        merge_cpu(arr, l, m, r, temp);
    }
}

void check_result_cpu(const int* __restrict__ original, const int* __restrict__ sorted, int size) {
    int* reference = (int*)malloc(size * sizeof(int));
    int* temp = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) reference[i] = original[i];

    /*CPU timing per array
    CpuTimer cpu_timer;
    cpu_timer.start();
    */
    merge_sort_cpu_parallel(reference, 0, size-1, temp, 3);
    /*CPU timing per array    
    cpu_timer.stop("CPU Merge Sort");
    */

    int errors = 0;
    for (int i = 0; i < size; ++i) {
        if (reference[i] != sorted[i]) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, reference[i], sorted[i]);
            errors++;
        }
    }
    if (errors == 0)
        printf("CPU-GPU validation passed.\n");
    else
        printf("CPU-GPU validation failed with %d mismatches.\n", errors);

    free(reference);
    free(temp);
}

int main() {
    print_device_info((const void*)sort_kernel, 256, 0);

    printf("Parameters\n");
    const int total_arrays = 32768;
    const int max_size = 1024;
    const int powers = 10;
    const int num_streams = 16;

    printf("Allocate host inputoutput\n");
    size_t max_bytes = max_size * sizeof(int);
    int *h_input, *h_output;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input,  total_arrays * max_bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, total_arrays * max_bytes));
    int *h_sizes  = (int*)malloc(total_arrays * sizeof(int));

    printf("Initialize sizes and data\n");
    for (int i = 0; i < total_arrays; ++i) {
        int pow = 3 + (i % powers);        
        int sz = 1 << pow;
	if(sz > 1024) sz=1024;
        h_sizes[i] = sz;
        int offset = i * max_size;
        for (int j = 0; j < sz; ++j) {
            h_input[offset + j] = rand();
        }
    }

    printf("Allocate device buffers per stream\n");
    int *d_buffers[num_streams];
    for (int s = 0; s < num_streams; ++s) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_buffers[s], max_bytes));
    }

    printf("Create streams\n");
    cudaStream_t streams[num_streams];
    for (int s = 0; s < num_streams; ++s) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[s]));
    }

    printf("Launch sorting across streams in batches\n");
    Timer timer;
    timer.start();

    const int batch_size = 128;
    for (int base = 0; base < total_arrays; base += batch_size) {
        int limit = (base + batch_size < total_arrays) ? base + batch_size : total_arrays;
        for (int i = base; i < limit; ++i) {
            int sz = h_sizes[i];
            int sid = i % num_streams;
            int *d_buf = d_buffers[sid];
            int *h_src = h_input + i * max_size;
            int *h_dst = h_output + i * max_size;

	    /*debug
	    printf("i=%d, sz=%d, sid=%d\n", i, sz, sid);
	    */

            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf, h_src, sz * sizeof(int), cudaMemcpyHostToDevice, streams[sid]));
            
	    /*debug
	    cudaFuncAttributes attr;
	    cudaFuncGetAttributes(&attr, sort_kernel);
	    printf("sharedSizeBytes=%lu, maxDynamicShared=%lu\n", (unsigned long)(sz * sizeof(int)), (unsigned long)attr.maxDynamicSharedSizeBytes);
	    */

	    sort_kernel<<<1, 256, sz * sizeof(int), streams[sid]>>>(d_buf, sz);
            CHECK_CUDA_ERROR(cudaGetLastError());
	    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_dst, d_buf, sz * sizeof(int), cudaMemcpyDeviceToHost, streams[sid]));
        }
        
        /*
        for (int s = 0; s < num_streams; ++s) {
                CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[s]));
            }
        */
    }

    for (int s = 0; s < num_streams; ++s) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[s]));
    }
    timer.stop("GPU Batch Merge Sort");

    /*CPU timing per array
    printf("CPU validation on first few arrays\n");
    check_result_cpu(h_input, h_output, h_sizes[0]);
    *CPU timing per array*/

    CpuTimer cpu_timer;
    cpu_timer.start();
    #pragma omp parallel for
    for (int i = 0; i < total_arrays; ++i) {
        int sz = h_sizes[i];
        int* temp = (int*)malloc(sz * sizeof(int));
        merge_sort_cpu_parallel(h_input + i * max_size, 0, sz - 1, temp, 3);
        free(temp);
    }
    cpu_timer.stop("CPU Batch Merge Sort");


    printf("Cleanup\n");
    for (int s = 0; s < num_streams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_buffers[s]);
    }
    CHECK_CUDA_ERROR(cudaFreeHost(h_input));
    CHECK_CUDA_ERROR(cudaFreeHost(h_output));
    free(h_sizes);

    CHECK_CUDA_ERROR(cudaDeviceReset());
    printf("CUDA device reset.\n");

    return 0;
}