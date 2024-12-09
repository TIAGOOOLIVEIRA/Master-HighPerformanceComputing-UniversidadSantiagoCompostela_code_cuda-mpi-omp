#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
//#include <omp.h>

// CSC Matrix Data Structure
typedef struct {
    double *values;       // Non-zero values
    int *row_indices;     // Row indices of non-zero values
    int *col_pointers;    // Column pointers (start of each column in `values`)
    int nnz;              // Number of non-zero elements
    int size;             // Matrix size (n x n)
} CSCMatrix;

void spmv_csc(CSCMatrix *matrix, double *vec, double *result) {
    int size = matrix->size;

    // Initialize the result vector to zero
    //#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
    }

    // Perform the SpMV operation
    //#pragma omp parallel for
    for (int col = 0; col < size; col++) {
        for (int idx = matrix->col_pointers[col]; idx < matrix->col_pointers[col + 1]; idx++) {
            int row = matrix->row_indices[idx];
            //#pragma omp atomic
            result[row] += matrix->values[idx] * vec[col];
        }
    }
}