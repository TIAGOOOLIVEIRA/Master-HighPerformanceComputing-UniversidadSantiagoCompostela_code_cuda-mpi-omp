#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define CUSTOM_UNIVERSE_SIZE_VALUE 8
#define ERR_ARGS 1
#define ERR_IO   2
#define ERR_MEM  3

int determine_optimal_processes(int rows, int cols, int max_procs) {
    int total_elements = rows * cols;

    if (total_elements <= 1000) return 1;
    if (total_elements <= 10000) return (max_procs >= 2) ? 2 : 1;
    if (total_elements <= 100000) return (max_procs >= 4) ? 4 : 2;
    if (total_elements <= 1000000) return (max_procs >= 8) ? 8 : 4;
    return max_procs;
}

int main(int argc, char *argv[]) {
    int world_rank, world_size, effective_procs;
    int gsize[2], lsize[2], starts[2];
    int *submatrix = NULL;
    int i, sum = 0, global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int custom_universe_size = CUSTOM_UNIVERSE_SIZE_VALUE;

    if (argc < 4) {
        if (world_rank == 0)
            fprintf(stderr, "Usage: %s file rows cols\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, ERR_ARGS);
    }

    gsize[0] = atoi(argv[2]);
    gsize[1] = atoi(argv[3]);

    effective_procs = determine_optimal_processes(gsize[0], gsize[1], custom_universe_size);

    if (effective_procs > world_size) {
        if (world_rank == 0)
            fprintf(stderr, "ERROR: This run only has %d MPI processes, but %d required. Use mpirun -np %d ...\n",
                    world_size, effective_procs, effective_procs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //communicator with the first `effective_procs` ranks
    MPI_Comm effective_comm;
    MPI_Group world_group, effective_group;
    int *ranks_in_group = malloc(effective_procs * sizeof(int));
    for (int i = 0; i < effective_procs; ++i) ranks_in_group[i] = i;

    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, effective_procs, ranks_in_group, &effective_group);
    if (world_rank < effective_procs)
        MPI_Comm_create(MPI_COMM_WORLD, effective_group, &effective_comm);
    else
        effective_comm = MPI_COMM_NULL;

    free(ranks_in_group);

    if (effective_comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int eff_rank, eff_size;
    MPI_Comm_rank(effective_comm, &eff_rank);
    MPI_Comm_size(effective_comm, &eff_size);

    int rows_per_rank = gsize[0] / effective_procs;
    int remainder = gsize[0] % effective_procs;
    int local_rows = rows_per_rank + (eff_rank < remainder ? 1 : 0);

    lsize[0] = local_rows;
    lsize[1] = gsize[1];
    starts[0] = (eff_rank < remainder)
                ? eff_rank * (rows_per_rank + 1)
                : remainder * (rows_per_rank + 1) + (eff_rank - remainder) * rows_per_rank;
    starts[1] = 0;

    MPI_File fh;
    MPI_Datatype filetype;

    MPI_Type_create_subarray(2, gsize, lsize, starts, MPI_ORDER_C, MPI_INT, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_open(effective_comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    submatrix = (int *)malloc(lsize[0] * lsize[1] * sizeof(int));

    if (!submatrix) {
        fprintf(stderr, "Error allocating memory\n");
        MPI_Abort(effective_comm, ERR_MEM);
    }

    MPI_File_set_view(fh, 0, MPI_INT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, submatrix, lsize[0] * lsize[1], MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    for (i = 0; i < lsize[0] * lsize[1]; i++) {
        sum += submatrix[i];
    }

    MPI_Reduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, effective_comm);

    if (eff_rank == 0) {
        printf("Sum is %d\n", global_sum);
        printf("Effective MPI processes used: %d\n", effective_procs);
    }

    free(submatrix);
    MPI_Type_free(&filetype);
    MPI_Group_free(&effective_group);
    MPI_Group_free(&world_group);
    MPI_Comm_free(&effective_comm);
    MPI_Finalize();
    return 0;
}
