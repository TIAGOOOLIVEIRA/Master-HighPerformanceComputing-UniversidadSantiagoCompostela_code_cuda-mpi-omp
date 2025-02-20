#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_cblas.h>      // CBLAS in GSL (the GNU Scientific Library)
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
//#include <gsl/gsl_spblas.h> 
#include "timer.h"
#include "spmv.h"
#include <omp.h>

#define DEFAULT_SIZE 16384
#define DEFAULT_DENSITY 0.1

//gcc `ls *.o` -lopenblas -fopenmp -o spmv
//ldd ./spmv
//./spmv
//
//gcc -O2 -ftree-vectorize -fstrict-aliasing -fopt-info-vec-optimized -fopt-info-vec=vec_report_gcc.txt spmv.c -c
//perf stat ./spmv


unsigned int populate_sparse_matrix(double mat[], unsigned int n, double density, unsigned int seed)
{
    unsigned int nnz = 0;

    #pragma omp parallel 
    {
        unsigned int local_nnz = 0;
        unsigned int thread_seed = seed + omp_get_thread_num();

        #pragma omp for
        for (unsigned int i = 0; i < n * n; i++) {
            if ((rand_r(&thread_seed) % 100) / 100.0 < density) {
                //pseudorandom value between -9.99 and 9.99
                mat[i] = ((double)(rand_r(&thread_seed) % 10) + (double)rand_r(&thread_seed) / RAND_MAX) * 
                         (rand_r(&thread_seed) % 2 == 0 ? 1 : -1);
                local_nnz++;
            } else {
                mat[i] = 0;
            }
        }

        #pragma omp atomic
        nnz += local_nnz;
    }

    return nnz;
}

unsigned int populate_vector(double vec[], unsigned int size, unsigned int seed)
{
  srand(seed);

  for (unsigned int i = 0; i < size; i++) {
    vec[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
  }

  return size;
}

int is_nearly_equal(double x, double y)
{
  const double epsilon = 1e-5 /* some small number */;
  return fabs(x - y) <= epsilon * fabs(x);
  // see Knuth section 4.2.2 pages 217-218
}

unsigned int check_result(double ref[], double result[], unsigned int size)
{
  for(unsigned int i = 0; i < size; i++) {
    if (!is_nearly_equal(ref[i], result[i]))
      return 0;
  }

  return 1;
}

int main(int argc, char *argv[])
{
  int size;        // number of rows and cols (size x size matrix)
  double density;  // aprox. ratio of non-zero values

  if (argc < 2) {
    size = DEFAULT_SIZE;
    density = DEFAULT_DENSITY;
  } else if (argc < 3) {
    size = atoi(argv[1]);
    density = DEFAULT_DENSITY;
  } else {
    size = atoi(argv[1]);
    density = (double) atoi(argv[2]) / 100.0;
  }

  double *mat, *vec, *refsol, *mysol;

  mat = (double *) malloc(size * size * sizeof(double));
  vec = (double *) malloc(size * sizeof(double));
  refsol = (double *) malloc(size * sizeof(double));
  mysol = (double *) malloc(size * sizeof(double));

  unsigned int nnz = populate_sparse_matrix(mat, size, density, 1);
  populate_vector(vec, size, 2);

  printf("Matriz size: %d x %d (%d elements)\n", size, size, size*size);
  printf("%d non-zero elements (%.2lf%%)\n\n", nnz, (double) nnz / (size*size) * 100.0);

  //
  // Dense computation using CBLAS (eg. GSL's CBLAS implementation)
  //
  printf("Dense computation\n----------------\n");

  timeinfo start, now;
  timestamp(&start);

  cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, mat, size, vec, 1, 0.0, refsol, 1);

  timestamp(&now);
  printf("Time taken by CBLAS dense computation: %ld ms\n", diff_milli(&start, &now));

  //
  // Using your own dense implementation
  //
  timestamp(&start);

  my_dense(size, mat, vec, mysol);

  timestamp(&now);
  printf("Time taken by my dense matrix-vector product: %ld ms\n", diff_milli(&start, &now));

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok!\n");
  else
    printf("Result is wrong!\n");


    //
    // Sparse computation using your own implementation
    //
    printf("\nSparse computation\n------------------\n");

    // Time the conversion to GSL
    timestamp(&start);

    // Convert dense matrix and vector to GSL structures
    GSLData gsl_data = convert_to_gsl(size, mat, vec);

    timestamp(&now);
    printf("Time taken by convert_to_gsl (Ref table 1:gsl-sparse): %ld ms\n", diff_milli(&start, &now));

    // Time the computation
    timestamp(&start);

    // Perform sparse matrix-vector multiplication
    compute_sparse(size, gsl_data, mysol);

    timestamp(&now);
    printf("Time taken by compute_sparse (Ref table 1:gsl-matmul): %ld ms\n", diff_milli(&start, &now));

    // Validate the result
    if (check_result(refsol, mysol, size) == 1) {
        printf("Result is correct for my_sparse!\n");
    } else {
        printf("Result is incorrect for my_sparse!\n");
    }

    // Sparse computation using CSR solver
    //
    printf("\nCSR Sparse computation\n------------------\n");

    // Convert dense matrix to CSR format
    timestamp(&start);
    CSRMatrix csr = convert_to_csr(mat, size);

    timestamp(&now);
    printf("Time taken by conversion to CSR computation: %ld ms\n", diff_milli(&start, &now));


    // Measure time for CSR solver
    timestamp(&start);

    my_csr(&csr, vec, mysol);

    timestamp(&now);
    printf("Time taken by CSR sparse computation: %ld ms\n", diff_milli(&start, &now));

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok for CSR sparse!\n");
  else
    printf("Result is wrong for CSR sparse!\n");


    //
    // COO Sparse computation
    //
    printf("\nCOO Sparse computation\n------------------\n");

    // Convert dense matrix to COO format
    timestamp(&start);
    COOMatrix coo = convert_to_coo(mat, size);
    timestamp(&now);
    printf("Time taken by conversion to COO: %ld ms\n", diff_milli(&start, &now));

    // Measure time for COO solver
    timestamp(&start);
    spmv_coo(&coo, vec, mysol);
    timestamp(&now);
    printf("Time taken by COO sparse computation: %ld ms\n", diff_milli(&start, &now));

    if (check_result(refsol, mysol, size) == 1)
        printf("Result is ok for COO sparse!\n");
    else
        printf("Result is wrong for COO sparse!\n");


    //
    // CSC Sparse computation
    //
    printf("\nCSC Sparse computation\n------------------\n");

    // Convert dense matrix to CSC format
    timestamp(&start);
    CSCMatrix csc = convert_to_csc(mat, size);
    timestamp(&now);
    printf("Time taken by conversion to CSC: %ld ms\n", diff_milli(&start, &now));

    // Measure time for CSC solver
    timestamp(&start);
    spmv_csc(&csc, vec, mysol);
    timestamp(&now);
    printf("Time taken by CSC sparse computation: %ld ms\n", diff_milli(&start, &now));

    if (check_result(refsol, mysol, size) == 1)
        printf("Result is ok for CSC sparse!\n");
    else
        printf("Result is wrong for CSC sparse!\n");


  free(mat);
  free(vec);
  free(refsol);
  free(mysol);

  //free csr
  free(csr.values);
  free(csr.col_indices);
  free(csr.row_ptr);

  //free COO
  free(coo.row_indices);
  free(coo.col_indices);
  free(coo.values);

  //free CSC
  free(csc.values);
  free(csc.row_indices);
  free(csc.col_pointers);

  //free gsl
  gsl_spmatrix_free(gsl_data.spmat);
  gsl_vector_free(gsl_data.gsl_vec);

  //free mkl
  /*
  free(mklCsr.values);
  free(mklCsr.col_indices);
  free(mklCsr.row_ptr);
  */

  return 0;
}