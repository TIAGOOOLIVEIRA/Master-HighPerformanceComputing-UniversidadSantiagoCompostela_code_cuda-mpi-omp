/* Dot product of two vectors */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "../../Lab1/shared/place_report_mpi.h"

#define MEC(call) { \
    int res = (call); \
    if (res != MPI_SUCCESS) { \
        char err_str[256]; int err_len; \
        MPI_Error_string(res, err_str, &err_len); \
        fprintf(stderr, "[Rank %d] MPI error at line %d: %s\n", rank, __LINE__, err_str); \
        MPI_Abort(MPI_COMM_WORLD, res); \
    } \
}

int main(int argc, char *argv[]) {
    int rank, size;
    double t_start, t_end;

    MEC(MPI_Init(&argc, &argv));
    MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    // Allocate as double to preserve precision
    double *x = (double *) malloc((end - start) * sizeof(double));
    double *y = (double *) malloc((end - start) * sizeof(double));
    if (!x || !y) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Initialize vectors
    for (long i = start; i < end; i++) {
        x[i - start] = (double)(N / 2.0 - i);
        y[i - start] = 0.0001 * i;
    }

    MEC(MPI_Barrier(MPI_COMM_WORLD));
    t_start = MPI_Wtime();

    double local_dot = 0.0;

    #pragma omp parallel for simd reduction(+:local_dot)
    for (long i = 0; i < (end - start); i++) {
        local_dot += x[i] * y[i];
    }

    printf("[Rank %d] Local dot product = %.10f (from indices %ld to %ld)\n",
           rank, local_dot, start, end - 1);

    double global_dot = 0.0;
    MPI_Request req;
    MEC(MPI_Ireduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req));
    MEC(MPI_Wait(&req, MPI_STATUS_IGNORE));

    MEC(MPI_Barrier(MPI_COMM_WORLD));
    t_end = MPI_Wtime();

    if (rank == 0) {
        printf("[Rank %d] Global dot product = %.10f\n", rank, global_dot);
        printf("[Rank %d] Execution time: %.6f seconds\n", rank, t_end - t_start);
    }

    free(x);
    free(y);
    MEC(MPI_Finalize());
    return 0;
}