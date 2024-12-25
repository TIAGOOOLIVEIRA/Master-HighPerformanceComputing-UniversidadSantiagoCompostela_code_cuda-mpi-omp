#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <likwid.h>

#define SIZE 1000 // Matrix size (SIZE x SIZE)


//to compile the code
//  gcc -fopenmp -O3 -march=native -o poc_malloc2d_vect poc_malloc2d_vect.c

//to assess with likwid
//  likwid-perfctr -C 0 -g FLOPS_DP ./simd_benchmark


// Function to allocate a 2D matrix dynamically
double **malloc2d(int rows, int cols) {
    double **matrix = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
    }
    return matrix;
}

// Function to free a 2D matrix
void free2d(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// OpenMP SIMD version for malloc2d
void test_malloc2d_simd(double **matrix) {
    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = i * j * 0.1;
        }
    }
}

// OpenMP SIMD version for flattened 1D matrix
void test_flattened_simd(double *matrix) {
    #pragma omp parallel for simd
    for (int i = 0; i < SIZE * SIZE; i++) {
        matrix[i] = (i / SIZE) * (i % SIZE) * 0.1;
    }
}

// Benchmark function for malloc2d
void benchmark_malloc2d() {
    double **matrix = malloc2d(SIZE, SIZE);

    clock_t start = clock();
    test_malloc2d_simd(matrix);
    clock_t end = clock();

    printf("malloc2d (SIMD) time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free2d(matrix, SIZE);
}

// Benchmark function for flattened 1D matrix
void benchmark_flattened() {
    double *matrix = malloc(SIZE * SIZE * sizeof(double));

    clock_t start = clock();
    test_flattened_simd(matrix);
    clock_t end = clock();

    printf("Flattened (SIMD) time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(matrix);
}

int main() {
    printf("Running SIMD benchmarks...\n");
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_REGISTER("malloc2d");
    LIKWID_MARKER_REGISTER("flattened");

    // Benchmark malloc2d
    LIKWID_MARKER_START("malloc2d");
    benchmark_malloc2d();
    LIKWID_MARKER_STOP("malloc2d");

    // Benchmark flattened
    LIKWID_MARKER_START("flattened");   
    benchmark_flattened();
    LIKWID_MARKER_STOP("flattened");

    LIKWID_MARKER_CLOSE;

    return 0;
}
