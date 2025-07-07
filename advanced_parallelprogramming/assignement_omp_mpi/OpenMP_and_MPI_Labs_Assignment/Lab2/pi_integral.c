#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "../Lab1/shared/place_report_mpi.h"

int main(int argc, char *argv[]) {
    int rank, size;
    long int n = 1000000000, i;
    double h, local_sum = 0.0, x, pi, global_sum;
    double PI25DT = 3.141592653589793238462643;
    double t_start, t_end;

    // --- Initialize MPI ---
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    place_report_mpi();

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s num_intervals\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    n = atol(argv[1]); 
    h = 1.0 / (double)n;

    // --- Timing start ---
    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    long int local_n = n / size;
    long int start = rank * local_n;
    long int end = (rank == size - 1) ? n : start + local_n;

    #pragma omp parallel for simd reduction(+:local_sum)
    for (i = start; i < end; i++) {
        x = h * ((double)i + 0.5);
        local_sum += 4.0 / (1.0 + x * x);
    }

    MPI_Request req;
    MPI_Ireduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &req);


    MPI_Wait(&req, MPI_STATUS_IGNORE);

    // --- Timing end ---
    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    if (rank == 0) {
        pi = h * global_sum;
        printf("[Rank %d] The obtained Pi value is: %.16f, the error is: %.16f\n",
               rank, pi, fabs(pi - PI25DT));
        printf("[Rank %d] Execution time: %.6f seconds\n", rank, t_end - t_start);
    }

    MPI_Finalize();
    return 0;
}
