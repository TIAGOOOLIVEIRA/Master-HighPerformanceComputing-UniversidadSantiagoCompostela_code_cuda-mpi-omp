#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>

// Function to convert dense matrix to CSR format
CSRMatrix convert_to_csr(const double *restrict mat, int size) {
    int nnz = 0;

    // Count non-zero elements
    for (int i = 0; i < size * size; i++) {
        if (mat[i] != 0.0) nnz++;
    }

    // Allocate CSR arrays
    CSRMatrix csr;
    csr.values = (double *)malloc(nnz * sizeof(double));
    csr.col_indices = (int *)malloc(nnz * sizeof(int));
    csr.row_ptr = (int *)malloc((size + 1) * sizeof(int));
    csr.nnz = nnz;
    csr.size = size;

    int value_index = 0;
    csr.row_ptr[0] = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double value = mat[i * size + j];
            if (value != 0.0) {
                csr.values[value_index] = value;
                csr.col_indices[value_index] = j;
                value_index++;
            }
        }
        csr.row_ptr[i + 1] = value_index;
    }

    return csr;
}

void my_csr(CSRMatrix *csr, const double *restrict vec, double *restrict result) {
    int size = csr->size;

    // Initialize result to zero
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
    }

    // Perform SpMV (Sparse Matrix-Vector Multiplication)
    for (int i = 0; i < size; i++) {
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++) {
            result[i] += csr->values[j] * vec[csr->col_indices[j]];
        }
    }
}