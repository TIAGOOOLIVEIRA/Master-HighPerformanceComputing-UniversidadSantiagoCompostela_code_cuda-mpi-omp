#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_spblas.h>
#include <omp.h>

//export OMP_NUM_THREADS=2
//gcc -fopenmp my_sparse.c -c
//

int my_sparse(const unsigned int n, const double* restrict mat, double* restrict vec, double* restrict result)
{
    //GSL sparse matrix in CSR format
    gsl_spmatrix *spmat = gsl_spmatrix_alloc(n, n);

    //Populate the sparse matrix with non-zero elements from the dense matrix
    //#pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            double value = mat[i * n + j];
            if (value != 0.0) {
                //#pragma omp critical
	        gsl_spmatrix_set(spmat, i, j, value);
            }
        }
    }

    //Create a GSL vector for the input and result vectors
    gsl_vector *gsl_vec = gsl_vector_alloc(n);
    gsl_vector *gsl_result = gsl_vector_alloc(n);

    //Populate the GSL vector with values from vec
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        gsl_vector_set(gsl_vec, i, vec[i]);
    }

    //Sparse matrix-vector multiplication: result = spmat * vec
    gsl_spblas_dgemv(CblasNoTrans, 1.0, spmat, gsl_vec, 0.0, gsl_result);

    //Result back to the output array
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        result[i] = gsl_vector_get(gsl_result, i);
    }

    gsl_spmatrix_free(spmat);
    gsl_vector_free(gsl_vec);
    gsl_vector_free(gsl_result); 
	
    return 0;
}
