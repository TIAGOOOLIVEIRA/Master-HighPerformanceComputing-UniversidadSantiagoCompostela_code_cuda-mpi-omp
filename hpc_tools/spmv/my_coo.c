#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

COOMatrix convert_to_coo(const double *restrict mat, int size) {
    int nnz = 0;

    // Count non-zero elements with OpenMP reduction
    #pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < size * size; i++) {
        if (mat[i] != 0.0) {
            nnz++;
        }
    }

    // Allocate COO components
    COOMatrix coo;
    coo.nnz = nnz;
    coo.size = size;
    coo.row_indices = (int *)malloc(nnz * sizeof(int));
    coo.col_indices = (int *)malloc(nnz * sizeof(int));
    coo.values = (double *)malloc(nnz * sizeof(double));

    // Parallel population of COO structure
    int *thread_counts, *thread_offsets;
    int num_threads = 1;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_counts = (int *)calloc(num_threads, sizeof(int));
            thread_offsets = (int *)calloc(num_threads, sizeof(int));
        }

        // Count non-zero elements per thread
        #pragma omp for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (mat[i * size + j] != 0.0) {
                    thread_counts[thread_id]++;
                }
            }
        }

        // Compute offsets for each thread
        #pragma omp single
        {
            int prefix_sum = 0;
            for (int t = 0; t < num_threads; t++) {
                thread_offsets[t] = prefix_sum;
                prefix_sum += thread_counts[t];
            }
        }

        // Populate COO format
        int local_idx = thread_offsets[thread_id];

        #pragma omp for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (mat[i * size + j] != 0.0) {
                    coo.row_indices[local_idx] = i;
                    coo.col_indices[local_idx] = j;
                    coo.values[local_idx] = mat[i * size + j];
                    local_idx++;
                }
            }
        }
    }

    free(thread_counts);
    free(thread_offsets);

    return coo;
}


void spmv_coo(const COOMatrix *restrict matrix, const double *restrict vec, double *result) {
    int size = matrix->size;

    // Initialize the result vector to zero
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
    }

    // Perform the SpMV operation
    for (int i = 0; i < matrix->nnz; i++) {
        int row = matrix->row_indices[i];
        double value = matrix->values[i];
        int col = matrix->col_indices[i];
        result[row] += value * vec[col];
    }
}