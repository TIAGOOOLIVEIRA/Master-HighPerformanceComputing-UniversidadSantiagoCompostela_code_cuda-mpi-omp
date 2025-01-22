#include "spmv_mkl.h"
#include <mkl.h>


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
        fprintf(stderr, "Error: Memory allocation failed for CSR components.\n");
        exit(EXIT_FAILURE);
    }

    // Populate CSR components
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
    if (!csr || !vec || !result) {
        fprintf(stderr, "Error: Null pointer passed to compute_sparse_mkl.\n");
        return;
    }

    sparse_matrix_t mkl_csr;
    struct matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT };

    // Check for errors during matrix creation
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

    // Release the MKL sparse matrix handle
    mkl_sparse_destroy(mkl_csr);
}
