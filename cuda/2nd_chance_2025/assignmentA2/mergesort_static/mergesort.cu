#include "../shared/helper.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define NUM_STREAMS 16
#define THREADS_PER_BLOCK 256
#define MAX_STATIC_SIZE 1024

__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end) {
    int i = start;
    int j = middle;
    for (int k = start; k < end; ++k) {
        if (i < middle && (j >= end || source[i] <= source[j])) {
            dest[k] = source[i++];
        } else {
            dest[k] = source[j++];
        }
    }
}

__global__ void sort_kernel_phase1(int* __restrict__ data, int size) {
    __shared__ int s_mem[2 * MAX_STATIC_SIZE];

    int* temp1 = s_mem;
    int* temp2 = s_mem + MAX_STATIC_SIZE;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < size; i += stride)
        temp1[i] = data[i];

    __syncthreads();

    for (int width = 1; width < size; width <<= 1) {
        for (int idx = tid * (width << 1); idx < size; idx += stride * (width << 1)) {
            int left = idx;
            int mid = min(idx + width, size);
            int right = min(idx + (width << 1), size);
            gpu_bottomUpMerge(temp1, temp2, left, mid, right);
        }
        __syncthreads();
        int* tmp = temp1;
        temp1 = temp2;
        temp2 = tmp;
        __syncthreads();
    }

    for (int i = tid; i < size; i += stride)
        data[i] = temp1[i];
}

void merge_sort_cpu_parallel(int* arr, int l, int r, int* temp, int depth) {
    if (l >= r) return;

    int m = (l + r) / 2;
    if (depth > 0) {
#pragma omp parallel sections
        {
#pragma omp section
            merge_sort_cpu_parallel(arr, l, m, temp, depth - 1);
#pragma omp section
            merge_sort_cpu_parallel(arr, m + 1, r, temp, depth - 1);
        }
    } else {
        merge_sort_cpu_parallel(arr, l, m, temp, depth - 1);
        merge_sort_cpu_parallel(arr, m + 1, r, temp, depth - 1);
    }

    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];

    for (i = l; i <= r; ++i)
        arr[i] = temp[i];
}

void write_array_to_file(const char* filename, int* array, int total_arrays, int max_size, int* sizes) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file %s\n", filename);
        return;
    }
    for (int i = 0; i < total_arrays; ++i) {
        for (int j = 0; j < sizes[i]; ++j) {
            fprintf(f, "%d ", array[i * max_size + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char** argv) {
    const int total_arrays = 32768;
    const int max_size = (argc > 1) ? atoi(argv[1]) : 1024;
    const int dump_files = (argc > 2) ? atoi(argv[2]) : 0;

    if (max_size > MAX_STATIC_SIZE) {
        printf("Unsupported max_size > %d\n", MAX_STATIC_SIZE);
        return -1;
    }

    print_device_info((const void*)sort_kernel_phase1, THREADS_PER_BLOCK, 0);

    printf("Default array size: %d\n", max_size);

    int* h_input;
    int* h_output;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input, total_arrays * max_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, total_arrays * max_size * sizeof(int)));

    int* h_sizes = (int*)malloc(total_arrays * sizeof(int));
    for (int i = 0; i < total_arrays; ++i) {
        int pow = 3 + (i % 10);
        int sz = 1 << pow;
        if (sz > max_size) sz = max_size;
        h_sizes[i] = sz;
        int offset = i * max_size;
        for (int j = 0; j < sz; ++j)
            h_input[offset + j] = rand();
    }

    if (dump_files) {
        write_array_to_file("input_array.txt", h_input, total_arrays, max_size, h_sizes);
    }

    int* d_buffers[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CHECK_CUDA_ERROR(cudaMalloc(&d_buffers[s], max_size * sizeof(int)));

    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s)
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[s]));

    Timer timer;
    timer.start();

    for (int base = 0; base < total_arrays; base += NUM_STREAMS) {
        int batch = min(NUM_STREAMS, total_arrays - base);

        for (int i = 0; i < batch; ++i) {
            int id = base + i;
            int sz = h_sizes[id];
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_buffers[i], h_input + id * max_size, sz * sizeof(int), cudaMemcpyHostToDevice, streams[i]));

            sort_kernel_phase1<<<1, THREADS_PER_BLOCK, 0, streams[i]>>>(d_buffers[i], sz);
        }

        for (int i = 0; i < batch; ++i) {
            int id = base + i;
            int sz = h_sizes[id];
            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output + id * max_size, d_buffers[i], sz * sizeof(int), cudaMemcpyDeviceToHost, streams[i]));
        }
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop("GPU Batch Merge Sort");

    if (dump_files) {
        write_array_to_file("gpu_sorted_array.txt", h_output, total_arrays, max_size, h_sizes);
    }

    CpuTimer cpu_timer;
    cpu_timer.start();
    int total_errors = 0;
#pragma omp parallel for reduction(+:total_errors)
    for (int i = 0; i < total_arrays; ++i) {
        int* ref = (int*)malloc(h_sizes[i] * sizeof(int));
        int* temp = (int*)malloc(h_sizes[i] * sizeof(int));
        for (int j = 0; j < h_sizes[i]; ++j)
            ref[j] = h_input[i * max_size + j];

        merge_sort_cpu_parallel(ref, 0, h_sizes[i]-1, temp, 3);

        for (int j = 0; j < h_sizes[i]; ++j) {
            if (ref[j] != h_output[i * max_size + j])
                total_errors++;
        }
        free(ref);
        free(temp);
    }
    cpu_timer.stop("CPU Batch Merge Sort");

    if (total_errors == 0)
        printf("CPU-GPU validation passed for all arrays.\n");
    else
        printf("CPU-GPU validation failed. %d total mismatches.\n", total_errors);

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaFree(d_buffers[s]);
        cudaStreamDestroy(streams[s]);
    }

    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    free(h_sizes);

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}