#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

CSCMatrix convert_to_csc(const double *restrict mat, int size) {
    int nnz = 0;

    // Count non-zero elements
    #pragma omp parallel for reduction(+:nnz)
    for (int i = 0; i < size * size; i++) {
        if (mat[i] != 0.0) {
            nnz++;
        }
    }

    // Allocate CSC components
    CSCMatrix csc;
    csc.nnz = nnz;
    csc.size = size;
    csc.values = (double *)malloc(nnz * sizeof(double));
    csc.row_indices = (int *)malloc(nnz * sizeof(int));
    csc.col_pointers = (int *)malloc((size + 1) * sizeof(int));

    // Populate CSC components
    int value_index = 0;
    csc.col_pointers[0] = 0;
    for (int col = 0; col < size; col++) {
        for (int row = 0; row < size; row++) {
            double value = mat[row * size + col];
            if (value != 0.0) {
                csc.values[value_index] = value;
                csc.row_indices[value_index] = row;
                value_index++;
            }
        }
        csc.col_pointers[col + 1] = value_index; // End of this column
    }

    return csc;
}

void spmv_csc(const CSCMatrix *restrict matrix, const double *restrict vec, double *restrict result) {
    int size = matrix->size;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
    }

    #pragma omp parallel for
    for (int col = 0; col < size; col++) {
        for (int idx = matrix->col_pointers[col]; idx < matrix->col_pointers[col + 1]; idx++) {
            int row = matrix->row_indices[idx];

            #pragma omp atomic
            result[row] += matrix->values[idx] * vec[col];
        }
    }
}