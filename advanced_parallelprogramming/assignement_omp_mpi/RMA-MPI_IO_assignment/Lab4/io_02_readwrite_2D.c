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

static void print_matrix(int * m, int h, int w);
static inline int f(int v)
{
  return v + 1;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  /* mpi_defs */
  int mpi_rank, mpi_size,
      psizes[2],            /* processes per dim */
      periods[2] = {1,1},   /* periodic grid */
      grid_rank,            /* local rank in grid */
      grid_coord[2];        /* process coordinates in grid */
  MPI_Comm grid_comm;       /* grid communicator */
  MPI_Datatype contig_t, colblock_t;

  /* other stuff */
  int gsizes[2] = {MATRIX_H, MATRIX_W}; /* global size */  
  int lsizes[2];                        /* local sizes */
  int *mat = 0,                         /* global matrix */
      *l_mat = 0;                       /* local matrix */

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc != 3)
  {
    if (!mpi_rank)
      printf("Usage: %s procs_y procs_x\n", argv[0]);
    MPI_Finalize();
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
    MPI_Finalize();
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
  MPI_Barrier(MPI_COMM_WORLD);

  /* create virtual topology */

  MPI_Cart_create(MPI_COMM_WORLD, 2, psizes, periods, 0, &grid_comm);
  MPI_Comm_rank(grid_comm, &grid_rank);
  MPI_Cart_coords(grid_comm, grid_rank, 2, grid_coord);


  MPI_Type_vector(lsizes[0], lsizes[1], gsizes[1], MPI_INT, &contig_t);
  MPI_Type_create_resized(contig_t,     /* input datatype */
                          0,            /* new lower bound */
                          sizeof(int),  /* new extent */
                          &colblock_t); /* new datatype (output) */
  MPI_Type_commit(&colblock_t);

  //TODO: Open the MPI file

  /* 1. Root process reads the input file */
  if (!mpi_rank)
  {
    FILE * fd;

    mat = (int *) malloc( gsizes[0] * gsizes[1] * sizeof(int) );
    fd = fopen(input_filename, "r");
    if (fread(mat, sizeof(int), gsizes[0] * gsizes[1], fd) != gsizes[0]*gsizes[1])
    {
      printf("Error reading input file or dimensions are wrong");
    }
    fclose(fd);
  }

  //TODO: Create a Distributed Array filetype with block distribution and set the File View

  /* 2. Data is scattered among all processes */

    int disps[psizes[0]*psizes[1]];
    int counts[psizes[0]*psizes[1]];
    for (int y=0; y<psizes[0]; y++) {
        for (int x=0; x<psizes[1]; x++) {
            disps[y*psizes[1]+x] = y*gsizes[1]*lsizes[0]+x*lsizes[1];
            counts [y*psizes[1]+x] = 1;
        }
    }

  //TODO: Read the file collectively

  l_mat = (int *) malloc( lsizes[0] * lsizes[1] * sizeof(int) );
  MPI_Scatterv(mat,   counts, disps, colblock_t,
              l_mat, lsizes[0] * lsizes[1], MPI_INT,
              0, grid_comm);

  for (int p=0; p<mpi_size; ++p)
  {
    if (mpi_rank == p)
    {
      printf("\nProcess {%d,%d}, local size =  %d x %d = %d:\n",
             grid_coord[0], grid_coord[1], lsizes[0], lsizes[1], lsizes[0] * lsizes[1]);
      print_matrix(l_mat, lsizes[0], lsizes[1]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /* 3. Apply function f to every element */
  for (int y=0; y<lsizes[0]; ++y)
    for (int x=0; x<lsizes[1]; ++x)
      l_mat[y*lsizes[1] + x] = f(l_mat[y*lsizes[1] + x]);

  //TODO: Open the output MPI file
  //TODO: Write the file collectively

  /* 4. Gather results back to root */
  MPI_Gatherv(l_mat, lsizes[0] * lsizes[1], MPI_INT,
              mat, counts, disps, colblock_t,
              0, grid_comm);

  /* 5. Print matrix to an output file */
  if (!mpi_rank)
  {
    /* print gathered matrix */
    printf("\nGathered output matrix:\n");
    print_matrix(mat, gsizes[0], gsizes[1]);

    FILE * fd = fopen(output_filename, "w");
    if (fwrite(mat, sizeof(int), gsizes[0] * gsizes[1], fd) != gsizes[0]*gsizes[1])
    {
      printf("Error writing output file or dimensions are wrong\n");
    }
    fclose(fd);
  }

  free(l_mat);
  if (!mpi_rank)
    free(mat);

  MPI_Finalize();

  return 0;
}

void print_matrix(int * m, int h, int w)
{
  /* print */
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