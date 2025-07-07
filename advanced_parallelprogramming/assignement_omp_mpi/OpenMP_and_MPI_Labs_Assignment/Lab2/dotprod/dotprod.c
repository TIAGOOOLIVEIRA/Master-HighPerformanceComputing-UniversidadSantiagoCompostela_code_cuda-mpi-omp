/* Dot product of two vectors */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "../../Lab1/shared/place_report_mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    double t_start, t_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    place_report_mpi();

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s num_elem_vector\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int N = atoi(argv[1]);
    long chunk = N / size;
    long start = rank * chunk;
    long end = (rank == size - 1) ? N : start + chunk;

    float *x = (float *) malloc((end - start) * sizeof(float));
    float *y = (float *) malloc((end - start) * sizeof(float));
    if (!x || !y) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //Initialize local slices
    for (long i = start; i < end; i++) {
        x[i - start] = (N / 2.0 - i);
        y[i - start] = 0.0001 * i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    double local_dot = 0.0;

    #pragma omp parallel for simd reduction(+:local_dot)
    for (long i = 0; i < (end - start); i++) {
        local_dot += x[i] * y[i];
    }

    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    if (rank == 0) {
        printf("[Rank %d] Dot product = %.10f\n", rank, global_dot);
        printf("[Rank %d] Execution time: %.6f seconds\n", rank, t_end - t_start);
    }

    free(x);
    free(y);
    MPI_Finalize();
    return 0;
}
