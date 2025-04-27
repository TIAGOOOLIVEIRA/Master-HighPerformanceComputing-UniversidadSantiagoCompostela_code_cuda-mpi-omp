#include "../shared/helper.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define MAX_STATIC_SIZE 1024

__global__ void sort_kernel(int* __restrict__ data, int size) {
    __shared__ int temp[MAX_STATIC_SIZE];
    __shared__ int temp2[MAX_STATIC_SIZE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < size; i += stride)
        temp[i] = data[i];
    __syncthreads();

    for (int width = 1; width < size; width <<= 1) {
        for (int idx = tid * (width << 1); idx < size; idx += stride * (width << 1)) {
            int left = idx;
            int mid = min(idx + width - 1, size - 1);
            int right = min(idx + (width << 1) - 1, size - 1);
            if (mid < right) {
                int l = left;
                int r = mid + 1;
                for (int k = left; k <= right; ++k) {
                    if (l <= mid && (r > right || temp[l] <= temp[r]))
                        temp2[k] = temp[l++];
                    else
                        temp2[k] = temp[r++];
                }
            }
        }
        __syncthreads();
        for (int i = tid; i < size; i += stride) {
            temp[i] = temp2[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < size; i += stride)
        data[i] = temp[i];
}

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

int check_result_cpu(const int* __restrict__ original, const int* __restrict__ sorted, int size) {
    int* reference = (int*)malloc(size * sizeof(int));
    int* temp = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) reference[i] = original[i];

    merge_sort_cpu_parallel(reference, 0, size-1, temp, 3);

    int errors = 0;
    for (int i = 0; i < size; ++i) {
        if (reference[i] != sorted[i]) {
            errors++;
        }
    }

    free(reference);
    free(temp);
    return errors;
}

int main(int argc, char** argv) {
    print_device_info((const void*)sort_kernel, 256, 0);

    const int total_arrays = 32768;
    const int max_size = (argc > 1) ? atoi(argv[1]) : 1024;
    const int num_streams = 16;

    if (max_size > MAX_STATIC_SIZE) {
        printf("Error: max_size %d exceeds static shared memory limit %d\n", max_size, MAX_STATIC_SIZE);
        return -1;
    }

    printf("Default array size: %d\n", max_size);
    printf("Total Mem footprint (input + output): %.2f MB\n", 2 * total_arrays * max_size * sizeof(int) / (1024.0 * 1024.0));

    printf("Allocate host input-output\n");
    size_t max_bytes = max_size * sizeof(int);
    int *h_input, *h_output;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input,  total_arrays * max_bytes));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, total_arrays * max_bytes));

    printf("Initialize sizes and data\n");
    int* h_sizes = (int*)malloc(total_arrays * sizeof(int));
    for (int i = 0; i < total_arrays; ++i) {
        int pow = 3 + (i % 10);
        int sz = 1 << pow;
        if (sz > max_size) sz = max_size;
        h_sizes[i] = sz;
        int offset = i * max_size;
        for (int j = 0; j < sz; ++j) {
            h_input[offset + j] = rand();
        }
    }

    printf("Allocate device buffers per stream\n");
    int* d_buffers[num_streams];
    for (int s = 0; s < num_streams; ++s) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_buffers[s], max_bytes));
    }

    printf("Create streams and events\n");
    cudaStream_t streams[num_streams];
    cudaEvent_t events[num_streams];
    for (int s = 0; s < num_streams; ++s) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[s]));
        CHECK_CUDA_ERROR(cudaEventCreate(&events[s]));
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

            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buf, h_src, sz * sizeof(int), cudaMemcpyHostToDevice, streams[sid]));

            int threads_per_block = 256;
            sort_kernel<<<1, threads_per_block, 0, streams[sid]>>>(d_buf, sz);

            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_dst, d_buf, sz * sizeof(int), cudaMemcpyDeviceToHost, streams[sid]));
            CHECK_CUDA_ERROR(cudaEventRecord(events[sid], streams[sid]));
        }
        for (int s = 0; s < num_streams; ++s) {
            CHECK_CUDA_ERROR(cudaEventSynchronize(events[s]));
        }
    }
    timer.stop("GPU Batch Merge Sort");

    printf("CPU validation on all arrays\n");
    Timer cpu_timer;
    cpu_timer.start();
    int total_errors = 0;
    #pragma omp parallel for reduction(+:total_errors)
    for (int i = 0; i < total_arrays; ++i) {
        total_errors += check_result_cpu(h_input + i * max_size, h_output + i * max_size, h_sizes[i]);
    }
    cpu_timer.stop("CPU Batch Merge Sort");

    if (total_errors == 0)
        printf("CPU-GPU validation passed for all arrays.\n");
    else
        printf("CPU-GPU validation failed. %d total mismatches.\n", total_errors);

    printf("Cleanup\n");
    for (int s = 0; s < num_streams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(events[s]);
        cudaFree(d_buffers[s]);
    }
    CHECK_CUDA_ERROR(cudaFreeHost(h_input));
    CHECK_CUDA_ERROR(cudaFreeHost(h_output));
    free(h_sizes);

    fflush(stdout);
    fflush(stderr);
    CHECK_CUDA_ERROR(cudaDeviceReset());
    printf("CUDA device reset.\n");

    omp_set_num_threads(1);
    exit(0);
}
