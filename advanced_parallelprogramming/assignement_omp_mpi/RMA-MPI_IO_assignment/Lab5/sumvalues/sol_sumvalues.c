#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ERR_ARGS 1
#define ERR_IO   2
#define ERR_MEM  3
#define ERR_READ 4

/* Custom communicator attribute */
int CUSTOM_UNIVERSE_SIZE_KEY;
#define CUSTOM_UNIVERSE_SIZE_VALUE 8

void __rsv_set_comm_attr();

int main(int argc, char *argv[]) {
    int mpi_rank, mpi_size;
    int gsize[2];
    int lsize[2];
    int *mat;
    FILE *input_file;
    int *displs = NULL, *send_counts = NULL;

    MPI_Init(&argc, &argv);

    // Set communicator attribute (must be done on all ranks)
    __rsv_set_comm_attr();

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // === Retrieve CUSTOM_UNIVERSE_SIZE attribute safely ===
    int *universe_ptr = NULL;
    int attr_found = 0;

    MPI_Attr_get(MPI_COMM_WORLD, CUSTOM_UNIVERSE_SIZE_KEY, &universe_ptr, &attr_found);

    if (!attr_found || universe_ptr == NULL) {
        if (mpi_rank == 0)
            fprintf(stderr, "Error: CUSTOM_UNIVERSE_SIZE attribute not found.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int custom_universe_size = *universe_ptr;
    if (mpi_rank == 0)
        printf("CUSTOM_UNIVERSE_SIZE = %d\n", custom_universe_size);

    if (argc < 4) {
        if (mpi_rank == 0)
            printf("Usage: %s file rows cols\n\n", argv[0]);
        MPI_Finalize();
        return ERR_ARGS;
    }

    gsize[0] = atoi(argv[2]);
    gsize[1] = atoi(argv[3]);

    // === Determine optimal process count (dummy logic based on row count) ===
    int desired_procs = gsize[0] / 100;  // example logic
    if (desired_procs < 1) desired_procs = 1;
    if (desired_procs > custom_universe_size) desired_procs = custom_universe_size;

    if (desired_procs != mpi_size && mpi_rank == 0) {
        printf("Warning: Requested %d MPI processes, but optimal logic suggests %d.\n",
               mpi_size, desired_procs);
    }

    lsize[0] = gsize[0] / mpi_size + (mpi_rank < gsize[0] % mpi_size ? 1 : 0);
    lsize[1] = gsize[1];

    if (mpi_rank == 0) {
        input_file = fopen(argv[1], "r");
        if (!input_file) {
            fprintf(stderr, "Error opening input file.\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_IO);
        }

        mat = malloc(sizeof(int) * gsize[0] * gsize[1]);
        if (!mat) {
            fprintf(stderr, "Error allocating memory on root.\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
        }

        if (fread(mat, sizeof(int), gsize[0] * gsize[1], input_file) != gsize[0] * gsize[1]) {
            fprintf(stderr, "Error reading matrix from input file.\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_READ);
        }

        fclose(input_file);

        displs = malloc(mpi_size * sizeof(int));
        send_counts = malloc(mpi_size * sizeof(int));

        displs[0] = 0;
        send_counts[0] = lsize[0] * lsize[1];
        for (int i = 1; i < mpi_size; i++) {
            send_counts[i] = (gsize[0] / mpi_size + (i < gsize[0] % mpi_size ? 1 : 0)) * lsize[1];
            displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    } else {
        mat = malloc(sizeof(int) * lsize[0] * lsize[1]);
        if (!mat) {
            fprintf(stderr, "Error allocating memory on rank %d.\n", mpi_rank);
            MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
        }
    }

    MPI_Scatterv(mat, send_counts, displs, MPI_INT,
                 mpi_rank ? mat : MPI_IN_PLACE, lsize[0] * lsize[1], MPI_INT,
                 0, MPI_COMM_WORLD);

    // === Local sum ===
    int local_sum = 0;
    for (int i = 0; i < lsize[0]; ++i)
        for (int j = 0; j < lsize[1]; ++j)
            local_sum += mat[i * lsize[1] + j];

    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
        printf("Sum is %d\n", global_sum);

    if (mat && mpi_rank == 0) free(mat);
    else if (mat) free(mat);
    if (displs) free(displs);
    if (send_counts) free(send_counts);

    MPI_Finalize();
    return 0;
}

void __rsv_set_comm_attr() {
    static int usv = CUSTOM_UNIVERSE_SIZE_VALUE;

    MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN,
                           &CUSTOM_UNIVERSE_SIZE_KEY, NULL);
    MPI_Comm_set_attr(MPI_COMM_WORLD, CUSTOM_UNIVERSE_SIZE_KEY, &usv);
}
