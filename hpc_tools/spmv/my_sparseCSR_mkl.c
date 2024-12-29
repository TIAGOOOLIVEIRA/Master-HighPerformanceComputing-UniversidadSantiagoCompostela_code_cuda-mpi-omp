//#include "spmv.h"
#include <mkl.h>

typedef struct {
    int *row_ptr;     // Row pointer array
    int *col_indices; // Column indices array
    double *values;   // Non-zero values array
    int nnz;          // Number of non-zero elements
    int size;         // Matrix size (n x n)
} MKLCSRMatrix;

MKLCSRMatrix convert_to_mkl_csr(const unsigned int n, const double *restrict mat) {
    MKLCSRMatrix csr;
    int nnz = 0;

    // Count non-zero elements
    for (unsigned int i = 0; i < n * n; i++) {
        if (mat[i] != 0.0) {
            nnz++;
        }
    }

    // Allocate CSR components
    csr.size = n;
    csr.nnz = nnz;
    csr.values = (double *)malloc(nnz * sizeof(double));
    csr.col_indices = (int *)malloc(nnz * sizeof(int));
    csr.row_ptr = (int *)malloc((n + 1) * sizeof(int));

    // Populate CSR components
    int value_index = 0;
    csr.row_ptr[0] = 0;
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            double value = mat[i * n + j];
            if (value != 0.0) {
                csr.values[value_index] = value;
                csr.col_indices[value_index] = j;
                value_index++;
            }
        }
        csr.row_ptr[i + 1] = value_index; // Row pointer
    }

    return csr;
}

void compute_sparse_mkl(const MKLCSRMatrix *csr, const double *restrict vec, double *restrict result) {
    // Create a sparse matrix handle for MKL
    sparse_matrix_t mkl_csr;
    matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT };

    // Initialize MKL's CSR matrix handle
    mkl_sparse_d_create_csr(&mkl_csr,
                            SPARSE_INDEX_BASE_ZERO,
                            csr->size,
                            csr->size,
                            csr->row_ptr,
                            csr->row_ptr + 1,
                            csr->col_indices,
                            csr->values);

    // Perform sparse matrix-vector multiplication: result = csr * vec
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0, // alpha
                    mkl_csr,
                    descr,
                    vec,
                    0.0, // beta
                    result);

    // Release the MKL sparse matrix handle
    mkl_sparse_destroy(mkl_csr);
}