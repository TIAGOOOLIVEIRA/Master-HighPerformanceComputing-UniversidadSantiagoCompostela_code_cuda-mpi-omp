#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define ERR_ARGS 1
#define ERR_IO   2
#define ERR_MEM  3
#define ERR_READ 4

#define CUSTOM_UNIVERSE_SIZE_VALUE 8
#define CRITICAL_ROWS_PER_PROC 100

void __rsv_set_comm_attr();
int decide_num_procs(int rows);

int main(int argc, char *argv[]) {
    int parent_rank;
    int universe_size = CUSTOM_UNIVERSE_SIZE_VALUE;
    MPI_Comm parent_comm, inter_comm;
    MPI_Init(&argc, &argv);

    MPI_Comm_get_parent(&parent_comm);
    //Parent process checks if it is the root process
    if (parent_comm == MPI_COMM_NULL) {
    
        if (argc < 4) {
            printf("Usage: %s file rows cols\n\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, ERR_ARGS);
        }

        char *filename = argv[1];
        int grows = atoi(argv[2]);
        int gcols = atoi(argv[3]);
        int total_elements = grows * gcols;

        int num_procs = decide_num_procs(grows);
        if (num_procs > universe_size) num_procs = universe_size;

        FILE *input_file = fopen(filename, "r");
        if (!input_file) {
            fprintf(stderr, "Error opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_IO);
        }
        int *matrix = malloc(total_elements * sizeof(int));
        if (!matrix) {
            fprintf(stderr, "Error allocating memory\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
        }
        if (fread(matrix, sizeof(int), total_elements, input_file) != total_elements) {
            fprintf(stderr, "Error reading input file\n");
            MPI_Abort(MPI_COMM_WORLD, ERR_READ);
        }
        fclose(input_file);

        //Data distribution - evenly
        int *send_counts = malloc(num_procs * sizeof(int));
        int *displs = malloc(num_procs * sizeof(int));
        int base_rows = grows / num_procs;
        int remaining = grows % num_procs;
        displs[0] = 0;
        for (int i = 0; i < num_procs; i++) {
            send_counts[i] = (base_rows + (i < remaining ? 1 : 0)) * gcols;
            if (i > 0)
                displs[i] = displs[i - 1] + send_counts[i - 1];
        }

        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Comm_spawn(argv[0], argv, num_procs, info, 0, MPI_COMM_SELF, &inter_comm, MPI_ERRCODES_IGNORE);

	printf("Parent: spawning child...\n");

        //Metadata and chunks
        for (int i = 0; i < num_procs; i++) {
            MPI_Send(&send_counts[i], 1, MPI_INT, i, 0, inter_comm);
            MPI_Send(&gcols, 1, MPI_INT, i, 1, inter_comm);
            MPI_Send(matrix + displs[i], send_counts[i], MPI_INT, i, 2, inter_comm);
        }

        int global_sum = 0, partial_sum;
        for (int i = 0; i < num_procs; i++) {
            MPI_Recv(&partial_sum, 1, MPI_INT, i, 3, inter_comm, MPI_STATUS_IGNORE);
            global_sum += partial_sum;
        }
        printf("Total sum: %d\n", global_sum);

        free(matrix);
        free(send_counts);
        free(displs);
        MPI_Info_free(&info);
    } else {
        //Spawned child
        MPI_Comm_rank(MPI_COMM_WORLD, &parent_rank);

	printf("Child: I'm the spawned process.\n");

        int recv_count, cols;
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
        MPI_Recv(&cols, 1, MPI_INT, 0, 1, parent_comm, MPI_STATUS_IGNORE);

        int *submat = malloc(recv_count * sizeof(int));
        if (!submat) MPI_Abort(MPI_COMM_WORLD, ERR_MEM);
        MPI_Recv(submat, recv_count, MPI_INT, 0, 2, parent_comm, MPI_STATUS_IGNORE);

        int local_sum = 0;
        for (int i = 0; i < recv_count; i++){
            local_sum += submat[i];
        }
            

        MPI_Send(&local_sum, 1, MPI_INT, 0, 3, parent_comm);
        free(submat);
    }

    MPI_Finalize();
    return 0;
}

int decide_num_procs(int rows) {
    return (rows + CRITICAL_ROWS_PER_PROC - 1) / CRITICAL_ROWS_PER_PROC;
}