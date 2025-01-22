typedef struct {
    int *row_ptr;     // Row pointer array
    int *col_indices; // Column indices array
    double *values;   // Non-zero values array
    int nnz;          // Number of non-zero elements
    int size;         // Matrix size (n x n)
} MKLCSRMatrix;

MKLCSRMatrix convert_to_mkl_csr(const unsigned int n, const double *restrict mat);
void compute_sparse_mkl(const MKLCSRMatrix *csr, const double *restrict vec, double *restrict result);