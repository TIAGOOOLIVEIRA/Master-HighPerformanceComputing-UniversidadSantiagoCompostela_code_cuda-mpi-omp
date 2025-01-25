#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include <gsl/gsl_cblas.h>
#include "spmv_mkl.h"

//to compile this file use the following command
//module load intel imkl
//icc -O3 spmv_mkl.c my_sparseCSR_mkl.c timer.c -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -o spmv_mkl

#define DEFAULT_SIZE 16384
#define DEFAULT_DENSITY 0.1


unsigned int populate_sparse_matrix(double mat[], unsigned int n, double density, unsigned int seed)
{
  unsigned int nnz = 0;

  srand(seed);

  for (unsigned int i = 0; i < n * n; i++) {
    if ((rand() % 100) / 100.0 < density) {
      // Get a pseudorandom value between -9.99 e 9.99
      mat[i] = ((double)(rand() % 10) + (double)rand() / RAND_MAX) * (rand() % 2 == 0 ? 1 : -1);
      nnz++;
    } else {
      mat[i] = 0;
    }
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
  timeinfo start, now;
  //mkl_set_num_threads(1);

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

  timestamp(&start);

  cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, mat, size, vec, 1, 0.0, refsol, 1);

  timestamp(&now);
  printf("Time taken by CBLAS dense computation: %ld ms\n", diff_milli(&start, &now));



    // Convert dense matrix to MKL CSR format
    timestamp(&start);
    MKLCSRMatrix mklCsr = convert_to_mkl_csr(size, mat);
    timestamp(&now);
    printf("Time taken by convert_to_mkl_csr (Ref table 2:mkl-sparse): %ld ms\n", diff_milli(&start, &now));

    // Perform sparse matrix-vector multiplication
    timestamp(&start);
    compute_sparse_mkl(&mklCsr, vec, mysol);
    timestamp(&now);
    printf("Time taken by compute_sparse_mkl (Ref table 2: mkl-sparse): %ld ms\n", diff_milli(&start, &now));

    // Validate the result
    if (check_result(refsol, mysol, size) == 1) {
        printf("Result is correct for MKL sparse computation!\n");
    } else {
        printf("Result is incorrect for MKL sparse computation!\n");
    }
    




  free(mat);
  free(vec);
  free(refsol);
  free(mysol);


  //free mkl
  
  free(mklCsr.values);
  free(mklCsr.col_indices);
  free(mklCsr.row_ptr);
  

  return 0;
}
