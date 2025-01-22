#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 512  // Matrix size

//matrix multiplication in ijk order
void matmul_ijk(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//matrix multiplication in ikj order
void matmul_ikj(double **A, double **B, double **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

//to allocate memory for a matrix
double **allocate_matrix(int size) {
    double **matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double *)malloc(size * sizeof(double));
    }
    return matrix;
}

//to initialize a matrix with random values
void initialize_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 10; 
        }
    }
}

//to free memory allocated for a matrix
void free_matrix(double **matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    srand(time(NULL));

    double **A = allocate_matrix(SIZE);
    double **B = allocate_matrix(SIZE);
    double **C = allocate_matrix(SIZE);
    double **D = allocate_matrix(SIZE);

    initialize_matrix(A, SIZE);
    initialize_matrix(B, SIZE);

    //execution time for ijk order
    clock_t start = clock();
    matmul_ijk(A, B, C, SIZE);
    clock_t end = clock();
    printf("Time taken for ijk implementation: %.3f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    //execution time for ikj order
    start = clock();
    matmul_ikj(A, B, D, SIZE);
    end = clock();
    printf("Time taken for ikj implementation: %.3f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free_matrix(A, SIZE);
    free_matrix(B, SIZE);
    free_matrix(C, SIZE);
    free_matrix(D, SIZE);

    return 0;
}