#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

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
    int rank, size, i, j, N, M;
    float **A = NULL, *Avector = NULL, *x = NULL, *y_local = NULL, *y_global = NULL;
    float temp;
    int local_rows, start_row, end_row;
    double t_start, t_end;

    MEC(MPI_Init(&argc, &argv));
    MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s N (assuming NxN matrix)\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    N = M = atoi(argv[1]);
    local_rows = N / size;
    start_row = rank * local_rows;
    end_row = (rank == size - 1) ? N : start_row + local_rows;

    Avector = (float *) malloc((end_row - start_row) * M * sizeof(float));
    A = (float **) malloc((end_row - start_row) * sizeof(float *));
    for (i = 0; i < (end_row - start_row); i++)
        A[i] = Avector + i * M;

    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < M; j++) {
            A[i - start_row][j] = (0.15 * i - 0.1 * j) / N;
        }
    }

    x = (float *) malloc(M * sizeof(float));
    y_local = (float *) malloc((end_row - start_row) * sizeof(float));
    if (rank == 0) {
        for (i = 0; i < M; i++)
            x[i] = (M / 2.0 - i);
    }

    MEC(MPI_Bcast(x, M, MPI_FLOAT, 0, MPI_COMM_WORLD));

    // --- Timing start ---
    MEC(MPI_Barrier(MPI_COMM_WORLD));
    t_start = MPI_Wtime();

    #pragma omp parallel for private(i, j, temp) shared(A, x, y_local)
    for (i = 0; i < (end_row - start_row); i++) {
        temp = 0.0;
        for (j = 0; j < M; j++)
            temp += A[i][j] * x[j];
        y_local[i] = temp;
    }

    if (rank == 0)
        y_global = (float *) malloc(N * sizeof(float));

    MEC(MPI_Gather(y_local, end_row - start_row, MPI_FLOAT,
                   y_global, end_row - start_row, MPI_FLOAT,
                   0, MPI_COMM_WORLD));

    // --- Timing end ---
    MEC(MPI_Barrier(MPI_COMM_WORLD));
    t_end = MPI_Wtime();

    float partial_sum = 0.0;
    for (i = 0; i < (end_row - start_row); i++) {
        partial_sum += y_local[i];
    }
    printf("[Rank %d] Local accumulated y = %f\n", rank, partial_sum);

    if (rank == 0) {
        printf("[Rank 0] Done, y[0] = %g  y[%d] = %g\n", y_global[0], N - 1, y_global[N - 1]);
        printf("[Rank 0] Total execution time: %.6f seconds\n", t_end - t_start);
    }

    free(Avector);
    free(A);
    free(x);
    free(y_local);
    if (rank == 0) free(y_global);

    MEC(MPI_Finalize());
    return 0;
}
