#include "spmv.h"
#include <stddef.h>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spblas.h> 
#include <omp.h>

GSLData convert_to_gsl(const unsigned int n, const double *restrict mat, const double *restrict vec) {
    GSLData data;

    data.spmat = gsl_spmatrix_alloc(n, n);

    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            double value = mat[i * n + j];
            if (value != 0.0) {
                #pragma omp critical 
                gsl_spmatrix_set(data.spmat, i, j, value);
            }
        }
    }

    data.gsl_vec = gsl_vector_alloc(n);
    
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        gsl_vector_set(data.gsl_vec, i, vec[i]);
    }

    return data;
}

void compute_sparse(const unsigned int n, GSLData data, double *restrict result) {
    gsl_vector *gsl_result = gsl_vector_alloc(n);

    gsl_spblas_dgemv(CblasNoTrans, 1.0, data.spmat, data.gsl_vec, 0.0, gsl_result);

    #pragma omp simd
    for (unsigned int i = 0; i < n; i++) {
        result[i] = gsl_vector_get(gsl_result, i);
    }

    gsl_vector_free(gsl_result);
}