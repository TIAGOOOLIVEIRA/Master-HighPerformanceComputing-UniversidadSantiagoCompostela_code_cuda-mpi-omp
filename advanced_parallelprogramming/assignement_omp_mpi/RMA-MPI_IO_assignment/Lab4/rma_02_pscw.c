/*
 * Example of active target synchronization using PSCW
 * 
 * Processes divide in 2 groups. Odd processes send data to even processes
 * 
 * Compile: mpicc -Wall -O3 -std=c99 -o 02_rma_pscw 02_rma_pscw.c
 * Run: no arguments required
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define A_SIZE 1000

int main(int argc, char ** argv)
{
  int *my_array;
  int size, rank;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //TODO: Create a Group out of the communicator with MPI_Comm_group
  //TODO: Include the odd/even processes in the group
  
  /* create private memory */
  //TODO: Create a memory window in "my_array",
  //      allocating memory at the same time
  my_array = (int *) malloc(size * sizeof(int));
  memset(my_array, 0, size*sizeof(int));

  if (rank % 2)
  {
    /* odd processes send data: 
     * we will place our rank in our partners' memory */
    int send_data = rank;
    
    //TODO: Enclose operations within an access epoch to the partner group
    
    //TODO: Replace these Two-Sided with One-Sided operations
    for (int i=0; i<size; i+=2)
      MPI_Send(&send_data, 1, MPI_INT,
               i, 0, MPI_COMM_WORLD);
  }
  else
  {
    /* even processes receive data */
    
    //TODO: Create an exposure epoch to the partner group
    
    //TODO: No operation call is needed in the target part
    for (int i=1; i<size; i+=2)
      MPI_Recv(my_array+i, 1, MPI_INT,
               i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
  }

  for (int i=0; i<size; i++)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (i == rank)
    {
      printf("Process %2d. A={ ", rank);
      for (int j=0; j<size; j++)
        if (my_array[j])
          printf("%2d ", my_array[j]);
        else
          printf(" - ");
      printf(" }\n");
    }
  }
  
  //TODO: Free MPI Window
  free(my_array);
    
  MPI_Finalize();
  return 0;
}

