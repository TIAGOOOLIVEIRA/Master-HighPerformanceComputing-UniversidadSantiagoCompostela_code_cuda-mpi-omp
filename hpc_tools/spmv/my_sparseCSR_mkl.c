#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "spmv_mkl.h"
#include <omp.h>


MKLCSRMatrix convert_to_mkl_csr(const unsigned int n, const double *restrict mat) {
    MKLCSRMatrix csr;
    csr.size = n;

    int *row_counts = (int *)calloc(n, sizeof(int));
    int nnz = 0;

    /*
    #pragma omp parallel for reduction(+:nnz)
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (mat[i * n + j] != 0.0) {
                row_counts[i]++;
                nnz++;
            }
        }
    }
    */

    int total_nnz = 0;

    #pragma omp parallel
    {
        int local_nnz = 0;
        #pragma omp for
        for (unsigned int i = 0; i < n; i++) {
            int count = 0;

            #pragma omp simd reduction(+:count)
            for (unsigned int j = 0; j < n; j++) {
                count += (mat[i * n + j] != 0.0);
            }

            row_counts[i] = count;
            local_nnz += count;
        }

        #pragma omp atomic
        total_nnz += local_nnz;
    }

    *nnz = total_nnz;


    csr.nnz = nnz;

    csr.values = (double *)malloc(nnz * sizeof(double));
    csr.col_indices = (int *)malloc(nnz * sizeof(int));
    csr.row_ptr = (int *)malloc((n + 1) * sizeof(int));

    if (!csr.values || !csr.col_indices || !csr.row_ptr) {
        printf("MeM allocation failed for CSR components.\n");
        exit(EXIT_FAILURE);
    }

    csr.row_ptr[0] = 0;
    for (unsigned int i = 0; i < n; i++) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + row_counts[i];
    }

    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        int start = csr.row_ptr[i];
        int offset = 0;

        for (unsigned int j = 0; j < n; j++) {
            if (mat[i * n + j] != 0.0) {
                csr.values[start + offset] = mat[i * n + j];
                csr.col_indices[start + offset] = j;
                offset++;
            }
        }
    }

    free(row_counts);

    return csr;
}


void compute_sparse_mkl(const MKLCSRMatrix *csr, const double *restrict vec, double *restrict result) {
    sparse_matrix_t mkl_csr;
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT };

    //Create MKL CSR
    if (mkl_sparse_d_create_csr(&mkl_csr,
                                 SPARSE_INDEX_BASE_ZERO,
                                 csr->size,
                                 csr->size,
                                 csr->row_ptr,
                                 csr->row_ptr + 1,
                                 csr->col_indices,
                                 csr->values) != SPARSE_STATUS_SUCCESS) {
        printf( "Error: Failed to create MKL CSR matrix.\n");
        return;
    }

    //To Validate CSR matrix
    printf("MKL CSR Matrix created successfully.\n");
    printf("Matrix size: %d\n", csr->size);
    printf("NNZ: %d\n", csr->nnz);

    // Debug: Print vectors
    //for (int i = 0; i < csr->size; i++) {
    //    printf("vec[%d] = %f\n", i, vec[i]);
    //}

    //Sparse mat-mul
    if (mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        1.0,
                        mkl_csr,
                        descr,
                        vec,
                        0.0,
                        result) != SPARSE_STATUS_SUCCESS) {
        printf("Error: MKL sparse matrix-vector multiplication failed.\n");
        mkl_sparse_destroy(mkl_csr);
        return;
    }

    printf("mkl_sparse_d_mv completed successfully\n");

    // Release MKL CSR matrix
    mkl_sparse_destroy(mkl_csr);
}