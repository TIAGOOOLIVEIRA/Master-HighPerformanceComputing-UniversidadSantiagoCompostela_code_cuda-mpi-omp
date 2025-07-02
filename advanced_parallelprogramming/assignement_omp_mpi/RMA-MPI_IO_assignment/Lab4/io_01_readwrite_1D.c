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



 /*
  References:
  - https://github.com/essentialsofparallelcomputing/Chapter16/blob/master/MPI_IO_Examples
  - Multicore and GPU Programming: An Integrated Approach, 2nd Edition
    by Gerassimos Barlas, 2023

 module load intel impi
 
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

#define V_PER_ROW 16
#define DO_PRINT   0

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

int f(int v)
{
  return v + 1;
}

void print_array(int * m, int w)
{
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

MPI_Offset file_offset = 0;

MPI_File create_mpi_io_file(const char *filename, MPI_Comm mpi_io_comm, long long file_size){
  int file_mode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_UNIQUE_OPEN;
  MPI_Info mpi_info = MPI_INFO_NULL;
  MEC(MPI_Info_create(&mpi_info));
  MEC(MPI_Info_set(mpi_info, "collective_buffering", "true"));
  MEC(MPI_Info_set(mpi_info, "striping_factor", "8"));
  MEC(MPI_Info_set(mpi_info, "striping_unit", "4194304"));

  MPI_File file_handle;
  MEC(MPI_File_open(mpi_io_comm, filename, file_mode, mpi_info, &file_handle));
  if (file_size > 0) MEC(MPI_File_set_size(file_handle, file_size));
  file_offset = 0;
  return file_handle;
}

MPI_File open_mpi_io_file(const char *filename, MPI_Comm mpi_io_comm){
  MPI_File file_handle;
  MEC(MPI_File_open(mpi_io_comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle));
  return file_handle;
}

void read_mpi_io_file(const char *filename, double **data, int data_size, MPI_Datatype memspace, MPI_Datatype filespace, MPI_Comm mpi_io_comm){
  MPI_File file_handle = open_mpi_io_file(filename, mpi_io_comm);

  MEC(MPI_File_set_view(file_handle, file_offset, MPI_DOUBLE, filespace, "native", MPI_INFO_NULL));
  MEC(MPI_File_read_all(file_handle, &(data[0][0]), 1, memspace, MPI_STATUS_IGNORE));
  file_offset += data_size;

  MEC(MPI_File_close(&file_handle));
  file_offset = 0;
}

void write_mpi_io_file(const char *filename, double **data, int data_size, MPI_Datatype memspace, MPI_Datatype filespace, MPI_Comm mpi_io_comm){
  MPI_File file_handle = create_mpi_io_file(filename, mpi_io_comm, (long long)data_size);

  MEC(MPI_File_set_view(file_handle, file_offset, MPI_DOUBLE, filespace, "native", MPI_INFO_NULL));
  MEC(MPI_File_write_all(file_handle, &(data[0][0]), 1, memspace, MPI_STATUS_IGNORE));
  file_offset += data_size;

  MEC(MPI_File_close(&file_handle));
  file_offset = 0;
}

int main(int argc, char **argv)
{
  MEC(MPI_Init(&argc, &argv));

  int mpi_rank, mpi_size;
  int *array = 0, *l_array = 0;
  char * input_filename  = argv[1];
  char * output_filename = argv[2];
  int gsize;
  int lsize;

  MEC(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MEC(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  if (argc != 3)
  {
    if (!mpi_rank)
      printf("Usage: %s input_fname output_fname\n", argv[0]);
    MEC(MPI_Finalize());
    exit(1);
  }

  if (!mpi_rank)
  {
    FILE * fd = fopen(input_filename, "r");
    fseek(fd, 0L, SEEK_END);
    long file_size = ftell(fd);
    fseek(fd, 0L, SEEK_SET);
    gsize = file_size / sizeof(int);

    array = (int *) malloc( gsize * sizeof(int) );
    if (fread(array, sizeof(int), gsize, fd) != gsize)
    {
      printf("Error reading input file or dimensions are wrong\n");
    }
    fclose(fd);
  }

  MEC(MPI_Bcast(&gsize, 1, MPI_INT, 0, MPI_COMM_WORLD));
  lsize = gsize/mpi_size;

  l_array = (int *) malloc( lsize * sizeof(int) );
  MEC(MPI_Scatter(array, lsize, MPI_INT, l_array, lsize, MPI_INT, 0, MPI_COMM_WORLD));

#if(DO_PRINT)
  for (int p=0; p<mpi_size; ++p)
  {
    if (mpi_rank == p)
    {
      printf("\nProcess %d/%d, local size =  %d\n", mpi_rank, mpi_size, lsize);
      print_array(l_array, lsize);
    }
    MEC(MPI_Barrier(MPI_COMM_WORLD));
  }
#endif

  MEC(MPI_Gather(l_array, lsize, MPI_INT, array, lsize, MPI_INT, 0, MPI_COMM_WORLD));

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

  MEC(MPI_Finalize());

  return 0;
}
