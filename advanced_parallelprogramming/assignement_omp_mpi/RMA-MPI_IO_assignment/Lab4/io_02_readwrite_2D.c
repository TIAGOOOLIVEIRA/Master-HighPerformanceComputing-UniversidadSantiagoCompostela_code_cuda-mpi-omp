/*
 * IO example 02: Simple centralized 2D I/O
 * using derived datatypes and virtual topology
 *
 * For simplicity, this program is hardcoded to work with data/integers.input
 * and a subset of MATRIX_H x MATRIX_W integers (input file actually contains many more).
 * 
 * Compile: mpicc -Wall -o io_02_readwrite_2D io_02_readwrite_2D.c
 *
 * Run arguments: Px Py, processes in each dimension
 *                Px times Py must equal the number of processes
 * e.g.: mpirun -np 4 ./io_02_readwrite_2D 2 2
 *       mpirun -np 4 ./io_02_readwrite_2D 1 4
 */


 #include <stdlib.h>
 #include <unistd.h>
 #include <stdio.h>
 #include <mpi.h>
 
 #define INPUT_FN  "data/integers.input"
 #define OUTPUT_FN "data/integers.2D.output"
 #define MATRIX_H 16
 #define MATRIX_W 16
 
 #define ERROR_ARGS 1
 #define ERROR_DIM  2
 
 #define MEC( call ) {int res; \
     res = call; \
     if (res != MPI_SUCCESS) { \
       char error_string[MPI_MAX_ERROR_STRING]; \
       int length_of_error_string; \
       MPI_Error_string(res, error_string, &length_of_error_string); \
       fprintf(stderr, "MPI Error: %s\n Call "#call"", error_string); \
       MPI_Abort(MPI_COMM_WORLD, res); \
     } \
     if (res == MPI_ERR_OTHER) { \
       fprintf(stderr, "An unknown error occurred.\n"); \
       MPI_Abort(MPI_COMM_WORLD, res); \
     } \
  }
 
 static void print_matrix(int * m, int h, int w);
 static inline int f(int v)
 {
   return v + 1;
 }
 
 int main(int argc, char **argv)
 {
   MEC(MPI_Init(&argc, &argv));
 
   int mpi_rank, mpi_size,
       psizes[2],
       periods[2] = {1,1},
       grid_rank,
       grid_coord[2];
   MPI_Comm grid_comm;
   MPI_Datatype contig_t, colblock_t, filetype;
 
   int gsizes[2] = {MATRIX_H, MATRIX_W};
   int lsizes[2];
   int *mat = 0, *l_mat = 0;
 
   MEC(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
   MEC(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
 
   if (argc != 3)
   {
     if (!mpi_rank)
       printf("Usage: %s procs_y procs_x\n", argv[0]);
     MEC(MPI_Finalize());
     exit(ERROR_ARGS);
   }
 
   const char * input_filename  = INPUT_FN;
   const char * output_filename = OUTPUT_FN;
   psizes[0] = atoi(argv[1]);
   psizes[1] = atoi(argv[2]);
 
   if (psizes[0] * psizes[1] != mpi_size)
   {
     if (!mpi_rank)
       printf("Error: procs_y (%d) x procs_x (%d) != mpi_size (%d)\n",
              psizes[0], psizes[1], mpi_size);
     MEC(MPI_Finalize());
     exit(ERROR_DIM);
   }
 
   lsizes[0] = gsizes[0] / psizes[0];
   lsizes[1] = gsizes[1] / psizes[1];
 
   if (!mpi_rank)
   {
     printf("Grid for %d processes in %d by %d grid\n",
            mpi_size, psizes[0], psizes[1]);
     printf("New datatype of %d elements with %d padding\n",
            lsizes[1], gsizes[1]);
   }
   MEC(MPI_Barrier(MPI_COMM_WORLD));
 
   MEC(MPI_Cart_create(MPI_COMM_WORLD, 2, psizes, periods, 0, &grid_comm));
   MEC(MPI_Comm_rank(grid_comm, &grid_rank));
   MEC(MPI_Cart_coords(grid_comm, grid_rank, 2, grid_coord));
 
   MEC(MPI_Type_vector(lsizes[0], lsizes[1], gsizes[1], MPI_INT, &contig_t));
   MEC(MPI_Type_create_resized(contig_t, 0, sizeof(int), &colblock_t));
   MEC(MPI_Type_commit(&colblock_t));
 
   int starts[2] = {grid_coord[0] * lsizes[0], grid_coord[1] * lsizes[1]};
   MEC(MPI_Type_create_subarray(2, gsizes, lsizes, starts, MPI_ORDER_C, MPI_INT, &filetype));
   MEC(MPI_Type_commit(&filetype));
 
   MPI_File input_fh;
   MEC(MPI_File_open(grid_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_fh));
   MEC(MPI_File_set_view(input_fh, 0, MPI_INT, filetype, "native", MPI_INFO_NULL));
 
   l_mat = (int *) malloc(lsizes[0] * lsizes[1] * sizeof(int));
   MEC(MPI_File_read_all(input_fh, l_mat, lsizes[0] * lsizes[1], MPI_INT, MPI_STATUS_IGNORE));
   MEC(MPI_File_close(&input_fh));
 
   for (int p=0; p<mpi_size; ++p)
   {
     if (mpi_rank == p)
     {
       printf("\nProcess {%d,%d}, local size =  %d x %d = %d:\n",
              grid_coord[0], grid_coord[1], lsizes[0], lsizes[1], lsizes[0] * lsizes[1]);
       print_matrix(l_mat, lsizes[0], lsizes[1]);
     }
     MEC(MPI_Barrier(MPI_COMM_WORLD));
   }
 
   for (int y=0; y<lsizes[0]; ++y)
     for (int x=0; x<lsizes[1]; ++x)
       l_mat[y*lsizes[1] + x] = f(l_mat[y*lsizes[1] + x]);
 
   if (!mpi_rank) mat = (int *) malloc(gsizes[0] * gsizes[1] * sizeof(int));
   int disps[psizes[0]*psizes[1]];
   int counts[psizes[0]*psizes[1]];
   for (int y=0; y<psizes[0]; y++) {
     for (int x=0; x<psizes[1]; x++) {
       disps[y*psizes[1]+x] = y*gsizes[1]*lsizes[0]+x*lsizes[1];
       counts[y*psizes[1]+x] = 1;
     }
   }
 
   MEC(MPI_Gatherv(l_mat, lsizes[0] * lsizes[1], MPI_INT,
                   mat, counts, disps, colblock_t,
                   0, grid_comm));
 
   if (!mpi_rank)
   {
     printf("\nGathered output matrix:\n");
     print_matrix(mat, gsizes[0], gsizes[1]);
   }
 
   MPI_File output_fh;
   MEC(MPI_File_open(grid_comm, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_fh));
   MEC(MPI_File_set_view(output_fh, 0, MPI_INT, filetype, "native", MPI_INFO_NULL));
   MEC(MPI_File_write_all(output_fh, l_mat, lsizes[0] * lsizes[1], MPI_INT, MPI_STATUS_IGNORE));
   MEC(MPI_File_close(&output_fh));
 
   free(l_mat);
   if (!mpi_rank) free(mat);
   MEC(MPI_Type_free(&colblock_t));
   MEC(MPI_Type_free(&filetype));
   MEC(MPI_Finalize());
   return 0;
 }
 
 void print_matrix(int * m, int h, int w)
 {
   for (int y=0; y<h; ++y)
   {
     for (int x=0; x<w; ++x)
     {
       printf("%4d ", m[y*w + x]);
     }
     printf("\n");
   }
   fflush(stdout);
 }
 