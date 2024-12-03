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

int my_dense(const unsigned int n, const double mat[], double vec[], double result[]);
int my_sparse(const unsigned int n, const double* restrict mat, double* restrict vec, double* restrict result);
void spmv_csc(CSCMatrix *matrix, double *vec, double *result);
void spmv_coo(COOMatrix *matrix, double *vec, double *result);
void my_csr(CSRMatrix *csr, double *restrict vec, double *restrict result);
CSRMatrix convert_to_csr(double *mat, int size);