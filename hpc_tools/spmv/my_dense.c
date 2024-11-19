#include "spmv.h"
#include <omp.h>

//for compiling with omp lib
//>>>>
//gcc -fopenmp my_dense.c -c
//gcc `ls *.o` -lopenblas -fopenmp -o spmv
//ldd ./spmv


int my_dense(const unsigned int n, const double mat[], double vec[], double result[])
{

    //omp to speedup the processing on the loop
    #pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) {
        result[i] = 0.0;
        //dot product
        for (unsigned int j = 0; j < n; j++) {
            result[i] += mat[i * n + j] * vec[j];
        }
    }
    return 0;  // Return 0 on success
}
