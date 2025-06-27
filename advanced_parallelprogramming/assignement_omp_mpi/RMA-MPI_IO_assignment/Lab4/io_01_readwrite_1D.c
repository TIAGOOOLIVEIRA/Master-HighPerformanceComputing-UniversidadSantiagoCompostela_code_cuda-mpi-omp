/*
 * IO example: Simple centralized I/O
 *
 * This example:
 * 1. Reads an array of integers from a file
 * 2. Scatters it among all processes
 * 4. Gathers the array back
 * 5. Prints the results to an output file
 *
 * Note that, input and output files have a binary format!
 *
 * Compile: mpicc -Wall -o io_01_readwrite_1D io_01_readwrite_1D.c
 * Run arguments: input_file output_file
 * e.g: mpirun -n N ./io_01_readwrite_1D data/integers.input data/integers.output
 * 
 * Disclaimer: For simplicity, many error checking statements
 *             are omitted.
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

#define V_PER_ROW 16
#define DO_PRINT   0

int f(int v)
{
  return v + 1;
}

void print_array(int * m, int w)
{
  /* print */
  for (int x=0; x<w; ++x)
  {
      printf("%7d ", m[x]);
      if (x && !(x % V_PER_ROW))
        printf("\n");
  }

  if (w % V_PER_ROW)
    printf("\n");

  fflush(stdout);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  /* mpi_defs */
  int mpi_rank, mpi_size;

  /* other stuff */
  int *array = 0,   /* global array */
      *l_array = 0; /* local array */

  char * input_filename  = argv[1];
  char * output_filename = argv[2];
  int gsize;
  int lsize;

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc != 3)
  {
    if (!mpi_rank)
      printf("Usage: %s input_fname output_fname\n", argv[0]);
    MPI_Finalize();
    exit(1);
  }

  //TODO: Replace step 1 with a collective File open.

  /* 1. Root process reads the input file */
  if (!mpi_rank)
  {
    FILE * fd = fopen(input_filename, "r");

    /* calculate the global size */
    fseek(fd, 0L, SEEK_END);
    long file_size = ftell(fd);
    fseek(fd, 0L, SEEK_SET);
    gsize = file_size / sizeof(int);

    /* read the input file */
    array = (int *) malloc( gsize * sizeof(int) );
    if (fread(array, sizeof(int), gsize, fd) != gsize)
    {
      printf("Error reading input file or dimensions are wrong\n");
    }
    fclose(fd);
  }

  MPI_Bcast(&gsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  lsize = gsize/mpi_size;

  //TODO: Replace step 2 with a collective read
  //      Set the file pointer to the beginning of each process' data
  
  /* 2. Data is scattered among all processes */
  l_array = (int *) malloc( lsize * sizeof(int) );
  MPI_Scatter(array,   lsize, MPI_INT,
              l_array, lsize, MPI_INT,
              0, MPI_COMM_WORLD);

#if(DO_PRINT)
  for (int p=0; p<mpi_size; ++p)
  {
    if (mpi_rank == p)
    {
      printf("\nProcess %d/%d, local size =  %d\n",
             mpi_rank, mpi_size, lsize);
      print_array(l_array, lsize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  //TODO: Replace steps 3 and 4 with MPI I/O operations (analogous to 1 and 2)
  //      but now, each process will define its own File View to perform a
  //      collective write.

  /* 3. Gather results back to root */
  MPI_Gather(l_array, lsize, MPI_INT,
             array,   lsize, MPI_INT,
             0, MPI_COMM_WORLD);

  /* 4. Print matrix to an output file */
  if (!mpi_rank)
  {
    FILE * fd = fopen(output_filename, "w");
    if (fwrite(array, sizeof(int), gsize, fd) != gsize)
    {
      printf("Error writing output file\n");
    }
    fclose(fd);
  }

  free(l_array);
  if (!mpi_rank)
    free(array);

  MPI_Finalize();

  return 0;
}
