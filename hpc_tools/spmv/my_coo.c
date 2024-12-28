#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>

COOMatrix convert_to_coo(const double *restrict mat, int size) {
    int nnz = 0;

    // Count non-zero elements
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

    // Populate COO components
    int idx = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (mat[i * size + j] != 0.0) {
                coo.row_indices[idx] = i;
                coo.col_indices[idx] = j;
                coo.values[idx] = mat[i * size + j];
                idx++;
            }
        }
    }

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