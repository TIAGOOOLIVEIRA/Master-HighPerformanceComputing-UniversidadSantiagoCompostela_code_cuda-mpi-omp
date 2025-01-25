#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include "spmv_mkl.h"


MKLCSRMatrix convert_to_mkl_csr(const unsigned int n, const double *restrict mat) {
    MKLCSRMatrix csr;
    csr.size = n;

    // Count non-zero elements
    int nnz = 0;
    for (unsigned int i = 0; i < n * n; i++) {
        if (mat[i] != 0.0) nnz++;
    }
    csr.nnz = nnz;

    // Allocate CSR components
    csr.values = (double *)malloc(nnz * sizeof(double));
    csr.col_indices = (int *)malloc(nnz * sizeof(int));
    csr.row_ptr = (int *)malloc((n + 1) * sizeof(int));

    if (!csr.values || !csr.col_indices || !csr.row_ptr) {
        printf("Error: Mem allocation failed for CSR.\n");
        exit(EXIT_FAILURE);
    }

    //Populate CSR
    int value_index = 0;
    csr.row_ptr[0] = 0;
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (mat[i * n + j] != 0.0) {
                csr.values[value_index] = mat[i * n + j];
                csr.col_indices[value_index] = j;
                value_index++;
            }
        }
        csr.row_ptr[i + 1] = value_index;
    }

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


void compute_sparse_mkl2(const MKLCSRMatrix *csr, const double *restrict vec, double *restrict result) {
    if (!csr || !vec || !result) {
        printf("Error: Null pointer passed to compute_sparse_mkl\n");
        return;
    }

    sparse_matrix_t mkl_csr;
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT };

    //Errors during matrix creation
    if (mkl_sparse_d_create_csr(&mkl_csr,
                                 SPARSE_INDEX_BASE_ZERO,
                                 csr->size,
                                 csr->size,
                                 csr->row_ptr,
                                 csr->row_ptr + 1,
                                 csr->col_indices,
                                 csr->values) != SPARSE_STATUS_SUCCESS) {
        printf("Error: Failed to create MKL CSR matrix.\n");
        return;
    }
    printf("mkl_sparse_d_create_csr\n");

    // Sparse matrix-vector multiplication: result = csr * vec
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
    printf("mkl_sparse_d_mv\n");

    mkl_sparse_destroy(mkl_csr);
}