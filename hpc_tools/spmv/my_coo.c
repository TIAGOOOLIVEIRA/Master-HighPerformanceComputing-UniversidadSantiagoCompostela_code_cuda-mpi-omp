#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
//#include <omp.h>

// COO Matrix Data Structure
typedef struct {
    double *values;  // Non-zero values
    int *row_indices;  // Row indices of non-zero values
    int *col_indices;  // Column indices of non-zero values
    int nnz;  // Number of non-zero elements
    int size;  // Matrix size (n x n)
} COOMatrix;

void spmv_coo(COOMatrix *matrix, double *vec, double *result) {
    int size = matrix->size;

    // Initialize the result vector to zero
    //#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
    }

    // Perform the SpMV operation
    //#pragma omp parallel for
    for (int i = 0; i < matrix->nnz; i++) {
        int row = matrix->row_indices[i];
        double value = matrix->values[i];
        int col = matrix->col_indices[i];
        //#pragma omp atomic
        result[row] += value * vec[col];
    }
}