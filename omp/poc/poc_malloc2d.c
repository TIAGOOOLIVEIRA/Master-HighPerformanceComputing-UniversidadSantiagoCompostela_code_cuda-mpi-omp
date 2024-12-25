#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <likwid.h>

#define SIZE 1000 // Size of the matrix (SIZE x SIZE)

//To use valgrind to check for memory leaks, compile the code with the -g flag: 
//    gcc -g poc_malloc2d.c -o poc_malloc2d
// to add likwid support 
//    gcc -o malloc2d_benchmark malloc2d_benchmark.c -llikwid
//To test cache miss rates via valgrind
//    valgrind --tool=cachegrind ./poc_malloc2d
//    valgrind --tool=cachegrind --I1=32768,8,64 --D1=32768,8,64 --LL=8388608,16,64 ./poc_malloc2d
//to performa other analysis via Likwid
//  likwid-perfctr -C 0 -g CACHE ./poc_malloc2d
//  likwid-perfctr -C 0 -g L2 -m ./poc_malloc2d
//  likwid-perfctr -C 0 -g MEM ./malloc2d_benchmark

//Not able to perform in MacOs due to the lack of support for perfctr in MacOs


// Allocate a 2D array using malloc2d
double **malloc2d(int rows, int cols) {
    double **array = malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        array[i] = malloc(cols * sizeof(double));
    }
    return array;
}

// Free 2D array
void free2d(double **array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void test_malloc2d() {
    double **matrix = malloc2d(SIZE, SIZE);

    // Fill and access the matrix
    clock_t start = clock();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = i * j * 0.1;
        }
    }
    clock_t end = clock();
    printf("malloc2d time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free2d(matrix, SIZE);
}

void test_flattened() {
    double *matrix = malloc(SIZE * SIZE * sizeof(double));

    // Fill and access the matrix
    clock_t start = clock();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i * SIZE + j] = i * j * 0.1;
        }
    }
    clock_t end = clock();
    printf("Flattened time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(matrix);
}

int main() {
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_REGISTER("malloc2d");
    LIKWID_MARKER_REGISTER("flattened");

    LIKWID_MARKER_START("malloc2d");
    test_malloc2d();
    LIKWID_MARKER_STOP("malloc2d");

    LIKWID_MARKER_START("flattened");    
    test_flattened();
    LIKWID_MARKER_STOP("flattened");
    
    LIKWID_MARKER_CLOSE;

    return 0;
}
