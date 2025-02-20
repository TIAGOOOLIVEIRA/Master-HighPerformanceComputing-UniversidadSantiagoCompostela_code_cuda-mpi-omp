#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function to convert dense matrix to CSR format
CSRMatrix convert_to_csr(const double *restrict mat, int size) {
    int nnz = 0;

    #pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < size * size; i++) {
        if (mat[i] != 0.0) {
            nnz++;
        }
    }

    CSRMatrix csr;
    csr.values = (double *)malloc(nnz * sizeof(double));
    csr.col_indices = (int *)malloc(nnz * sizeof(int));
    csr.row_ptr = (int *)malloc((size + 1) * sizeof(int));
    csr.nnz = nnz;
    csr.size = size;

    // Parallelized population using thread-local storage
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

        //Step1: Count non-zero elements per row per thread
        #pragma omp for
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (mat[i * size + j] != 0.0) {
                    thread_counts[thread_id]++;
                }
            }
        }

        //Step2: Compute offsets for each thread
        #pragma omp single
        {
            int prefix_sum = 0;
            for (int t = 0; t < num_threads; t++) {
                thread_offsets[t] = prefix_sum;
                prefix_sum += thread_counts[t];
            }
        }

        //Step3: Populate CSR structure in parallel
        int local_idx = thread_offsets[thread_id];

        #pragma omp for
        for (int i = 0; i < size; i++) {
            csr.row_ptr[i] = local_idx;

            for (int j = 0; j < size; j++) {
                double value = mat[i * size + j];
                if (value != 0.0) {
                    csr.values[local_idx] = value;
                    csr.col_indices[local_idx] = j;
                    local_idx++;
                }
            }
        }

        //Last row pointer is set correctly
        #pragma omp single
        {
            csr.row_ptr[size] = nnz;
        }
    }

    free(thread_counts);
    free(thread_offsets);

    return csr;
}

void my_csr(CSRMatrix *csr, const double *restrict vec, double *restrict result) {
    int size = csr->size;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
    }

    //SpMV with OpenMP and SIMD
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        #pragma omp simd
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++) {
            result[i] += csr->values[j] * vec[csr->col_indices[j]];
        }
    }
}