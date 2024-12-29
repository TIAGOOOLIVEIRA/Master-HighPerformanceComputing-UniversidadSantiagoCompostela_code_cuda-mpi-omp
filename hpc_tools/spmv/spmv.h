#include <gsl/gsl_spblas.h> 

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

typedef struct {
    gsl_spmatrix *spmat;
    gsl_vector *gsl_vec;
} GSLData;

int my_dense(const unsigned int n, const double mat[], double vec[], double result[]);
GSLData convert_to_gsl(const unsigned int n, const double *restrict mat, const double *restrict vec);
void compute_sparse(const unsigned int n, GSLData data, double result[]);
void spmv_csc(const CSCMatrix *restrict matrix, const double *restrict vec, double *result);
CSCMatrix convert_to_csc(const double *restrict mat, int size);
void spmv_coo(const COOMatrix *restrict matrix, const double *restrict vec, double *result);
COOMatrix convert_to_coo(const double *restrict mat, int size);
void my_csr(CSRMatrix *csr, const double *restrict vec, double *restrict result);
CSRMatrix convert_to_csr(const double *restrict mat, int size);