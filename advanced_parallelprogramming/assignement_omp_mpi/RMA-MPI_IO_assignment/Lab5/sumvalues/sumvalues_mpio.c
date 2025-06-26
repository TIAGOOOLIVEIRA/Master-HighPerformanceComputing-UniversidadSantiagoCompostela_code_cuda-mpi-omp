#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ERR_ARGS 1
#define ERR_IO   2
#define ERR_MEM  3
#define ERR_READ 4

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_File fh;
    MPI_Status status;
    int gsizes[2], lsizes[2], starts[2];
    int *local_matrix;
    int i, j, sum = 0, global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s filename rows cols\n", argv[0]);
        }
        MPI_Finalize();
        return ERR_ARGS;
    }

    const char *filename = argv[1];
    gsizes[0] = atoi(argv[2]);
    gsizes[1] = atoi(argv[3]);

    //Split rows among processes
    lsizes[0] = gsizes[0] / size;
    int remainder = gsizes[0] % size;
    if (rank < remainder) {
        lsizes[0]++;
    }
    lsizes[1] = gsizes[1];

    starts[0] = (gsizes[0] / size) * rank + (rank < remainder ? rank : remainder);
    starts[1] = 0;

    local_matrix = (int *)malloc(lsizes[0] * lsizes[1] * sizeof(int));
    if (!local_matrix) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
    }

    //File view for each process
    MPI_Datatype filetype;
    MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_C, MPI_INT, &filetype);
    MPI_Type_commit(&filetype);

    //read the matrix from the file
    if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: Unable to open file %s\n", rank, filename);
        MPI_Abort(MPI_COMM_WORLD, ERR_IO);
    }

    //File view and read
    MPI_File_set_view(fh, 0, MPI_INT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, local_matrix, lsizes[0] * lsizes[1], MPI_INT, &status);

    MPI_File_close(&fh);
    MPI_Type_free(&filetype);

    for (i = 0; i < lsizes[0]; ++i) {
        for (j = 0; j < lsizes[1]; ++j) {
            sum += local_matrix[i * lsizes[1] + j];
        }
    }

    MPI_Reduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum is %d\n", global_sum);
    }

    free(local_matrix);
    MPI_Finalize();
    return 0;
}