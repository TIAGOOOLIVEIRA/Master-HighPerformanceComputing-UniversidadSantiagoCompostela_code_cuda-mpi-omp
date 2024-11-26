int my_dense(const unsigned int n, const double mat[], double vec[], double result[]);
int my_sparse(const unsigned int n, const double mat[], double vec[], double result[]);
void spmv_csc(CSCMatrix *matrix, double *vec, double *result);
void spmv_coo(COOMatrix *matrix, double *vec, double *result);
