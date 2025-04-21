#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (code %d)\n", cudaGetErrorString(err), err); \
            exit(EXIT_FAILURE); \
        } \
    }

__device__ void merge(int* data, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (data[i] <= data[j]) temp[k++] = data[i++];
        else temp[k++] = data[j++];
    }
    while (i <= mid) temp[k++] = data[i++];
    while (j <= right) temp[k++] = data[j++];
    for (i = left; i <= right; i++) data[i] = temp[i];
}

__global__ void sort_kernel(int* data, int size) {
    extern __shared__ int temp[];
    for (int width = 1; width < size; width *= 2) {
        for (int i = 0; i < size; i += 2 * width) {
            int left = i;
            int mid = min(i + width - 1, size - 1);
            int right = min(i + 2 * width - 1, size - 1);
            if (mid < right)
                merge(data, temp, left, mid, right);
        }
    }
}

__global__ void merge_kernel(const int* left, int sizeL, const int* right, int sizeR, int* output) {
    int i = 0, j = 0, k = 0;
    while (i < sizeL && j < sizeR) {
        output[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    }
    while (i < sizeL) output[k++] = left[i++];
    while (j < sizeR) output[k++] = right[j++];
}

void merge_cpu(int* arr, int l, int m, int r, int* temp) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (i = l; i <= r; ++i) arr[i] = temp[i];
}

void merge_sort_cpu_parallel(int* arr, int l, int r, int* temp, int depth) {
    if (l < r) {
        int m = (l + r) / 2;
        if (depth <= 0) {
            merge_sort_cpu_parallel(arr, l, m, temp, depth - 1);
            merge_sort_cpu_parallel(arr, m + 1, r, temp, depth - 1);
        } else {
            #pragma omp parallel sections
            {
                #pragma omp section
                merge_sort_cpu_parallel(arr, l, m, temp, depth - 1);
                #pragma omp section
                merge_sort_cpu_parallel(arr, m + 1, r, temp, depth - 1);
            }
        }
        merge_cpu(arr, l, m, r, temp);
    }
}

void check_result_cpu(const int* original, const int* sorted, int size) {
    int* reference = (int*)malloc(size * sizeof(int));
    int* temp = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) reference[i] = original[i];

    merge_sort_cpu_parallel(reference, 0, size - 1, temp, 3);

    int errors = 0;
    for (int i = 0; i < size; i++) {
        if (reference[i] != sorted[i]) {
            printf("Mismatch at idx %d: CPU = %d, GPU = %d\n", i, reference[i], sorted[i]);
            errors++;
        }
    }

    if (errors == 0)
        printf("CPU-GPU checked OK\n");
    else
        printf("CPU-GPU N-OK, with %d mismatches.\n", errors);

    free(reference);
    free(temp);
}

int main() {
    const int N = 16;
    int h_input[N] = {20, 5, 3, 9, 1, 4, 7, 6, 14, 13, 11, 10, 2, 8, 12, 15};
    int *h_output;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, N * sizeof(int)));

    int mid = N / 2;
    int *d_left, *d_right, *d_merged;
    CHECK_CUDA_ERROR(cudaMalloc(&d_left, mid * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_right, (N - mid) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_merged, N * sizeof(int)));

    cudaStream_t s_left, s_right, s_merge;
    CHECK_CUDA_ERROR(cudaStreamCreate(&s_left));
    CHECK_CUDA_ERROR(cudaStreamCreate(&s_right));
    CHECK_CUDA_ERROR(cudaStreamCreate(&s_merge));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_left, h_input, mid * sizeof(int), cudaMemcpyHostToDevice, s_left));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_right, h_input + mid, (N - mid) * sizeof(int), cudaMemcpyHostToDevice, s_right));

    //to sort each half
    sort_kernel<<<1, 1, mid * sizeof(int), s_left>>>(d_left, mid);
    sort_kernel<<<1, 1, (N - mid) * sizeof(int), s_right>>>(d_right, N - mid);

    cudaEvent_t eL, eR;
    CHECK_CUDA_ERROR(cudaEventCreate(&eL));
    CHECK_CUDA_ERROR(cudaEventCreate(&eR));
    CHECK_CUDA_ERROR(cudaEventRecord(eL, s_left));
    CHECK_CUDA_ERROR(cudaEventRecord(eR, s_right));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(s_merge, eL, 0));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(s_merge, eR, 0));

    // Merge result
    merge_kernel<<<1, 1, 0, s_merge>>>(d_left, mid, d_right, N - mid, d_merged);

    // Copy back
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_merged, N * sizeof(int), cudaMemcpyDeviceToHost, s_merge));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(s_merge));

    printf("GPU Sorted: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_output[i]);
    printf("\n");

    check_result_cpu(h_input, h_output, N);

    // Cleanup
    cudaFreeHost(h_output);
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_merged);
    cudaStreamDestroy(s_left);
    cudaStreamDestroy(s_right);
    cudaStreamDestroy(s_merge);
    cudaEventDestroy(eL);
    cudaEventDestroy(eR);
 
    CHECK_CUDA_ERROR(cudaDeviceReset());
    printf("CUDA device reset.\n");

    return 0;
}