#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_cblas.h>      // CBLAS in GSL (the GNU Scientific Library)
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include "timer.h"
#include "spmv.h"

#define DEFAULT_SIZE 1024
#define DEFAULT_DENSITY 0.25

//export OMP_NUM_THREADS=2
//gcc `ls *.o` -lopenblas -fopenmp -o spmv
//ldd ./spmv
//./spmv
//
//gcc -O2 -ftree-vectorize -fstrict-aliasing -fopt-info-vec-optimized -fopt-info-vec=vec_report_gcc.txt spmv.c -c
//perf stat ./spmv

typedef struct {
    double *values;       // Non-zero values
    int *row_indices;     // Row indices of non-zero values
    int *col_pointers;    // Column pointers (start of each column in `values`)
    int nnz;              // Number of non-zero elements
    int size;             // Matrix size (n x n)
} CSCMatrix;

// COO Matrix Data Structure
typedef struct {
    double *values;  // Non-zero values
    int *row_indices;  // Row indices of non-zero values
    int *col_indices;  // Column indices of non-zero values
    int nnz;  // Number of non-zero elements
    int size;  // Matrix size (n x n)
} COOMatrix;

// CSR Matrix Data Structure
typedef struct {
    double *values;      // Non-zero values
    int *col_indices;    // Column indices of non-zero values
    int *row_ptr;        // Row pointer array
    int nnz;             // Number of non-zero elements
    int size;            // Matrix size (n x n)
} CSRMatrix;

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
  // Let's try now SpMV: Sparse Matrix - Dense Vector computation
  //

  // Convert mat to a sparse format: CSR
  // Use the gsl_spmatrix struct as datatype

  //
  // Sparse computation using GSL's sparse algebra functions
  //

  //
  // Your own sparse implementation
  //

  // Compare times (and computation correctness!)


  // Free resources
  timestamp(&start);
  my_sparse(size, mat, vec, mysol);
  timestamp(&now);
  printf("\n\nTime taken by my sparse matrix-vector product: %ld ms\n", diff_milli(&start, &now));

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok for sparse!\n");
  else
    printf("Result is wrong for sparse!\n");

//
    // Sparse computation using CSR solver
    //
    printf("\nCSR Sparse computation\n------------------\n");

    // Convert dense matrix to CSR format
    CSRMatrix csr = convert_to_csr(mat, size);

    // Measure time for CSR solver
    timestamp(&start);

    my_csr(&csr, vec, mysol);

    timestamp(&now);
    printf("Time taken by CSR sparse computation: %ld ms\n", diff_milli(&start, &now));

  if (check_result(refsol, mysol, size) == 1)
    printf("Result is ok for CSR sparse!\n");
  else
    printf("Result is wrong for CSR sparse!\n");

  free(mat);
  free(vec);
  free(refsol);
  free(mysol);
  free(csr.values);
  free(csr.col_indices);
  free(csr.row_ptr);

  return 0;
}
