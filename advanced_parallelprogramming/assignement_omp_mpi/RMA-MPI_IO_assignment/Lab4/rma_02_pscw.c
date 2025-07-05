/*
 * Example of active target synchronization using PSCW (Post-Start-Complete-Wait)
 *
 * Processes divide in 2 groups. Odd processes send data to even processes.
 * This implementation uses One-Sided Communication with Active Target Synchronization.
 *
 * Compile: mpicc -Wall -O3 -std=c99 -o rma_02_pscw rma_02_pscw.c
 * Run: No arguments required
 * *  mpirun -np 4 ./rma_02_pscw
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define A_SIZE 1000

#define MEC( call ) {int res; \
    res = call; \
    if (res != MPI_SUCCESS) { \
      char error_string[MPI_MAX_ERROR_STRING]; \
      int length_of_error_string; \
      MPI_Error_string(res, error_string, &length_of_error_string); \
      fprintf(stderr, "MPI Error: %s\n Call "#call"\n", error_string); \
      MPI_Abort(MPI_COMM_WORLD, res); \
    } \
}

int main(int argc, char ** argv)
{
  int *my_array;
  int size, rank;

  MEC(MPI_Init(&argc, &argv));
  MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  
  MPI_Group world_group, partner_group;
  MEC(MPI_Comm_group(MPI_COMM_WORLD, &world_group));

  //partner ranks list (odd <-> even communication)
  int partners[size/2], count = 0;
  for (int i = 0; i < size; i++) {
    if ((rank % 2 == 0 && i % 2 != 0) || (rank % 2 != 0 && i % 2 == 0)) {
      partners[count++] = i;
    }
  }

  MEC(MPI_Group_incl(world_group, count, partners, &partner_group));

  MPI_Win win;
  MEC(MPI_Win_allocate(size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD,
                       &my_array, &win));
  memset(my_array, 0, size * sizeof(int));  // zero initialization

  if (rank % 2 != 0) {
    //odd processes to send data using One-Sided Put
    int send_data = rank;

    //Post|Start-Complete-Wait (PSCW) synchronization
    //Post Access Epoch
    MEC(MPI_Win_start(partner_group, 0, win));

    for (int i = 0; i < count; i++) {
      int target_rank = partners[i];
      MEC(MPI_Put(&send_data, 1, MPI_INT,
                  target_rank, rank, 1, MPI_INT, win));
    }

    //end Access epoch
    MEC(MPI_Win_complete(win));
  }
  else {
    //even processes expose mem region to odd partners
    MEC(MPI_Win_post(partner_group, 0, win));
    MEC(MPI_Win_wait(win));
  }

  
  for (int i=0; i<size; i++)
  {
    MEC(MPI_Barrier(MPI_COMM_WORLD));
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
  
  MEC(MPI_Group_free(&partner_group));
  MEC(MPI_Group_free(&world_group));
  MEC(MPI_Win_free(&win));
  MEC(MPI_Finalize());

  return 0;
}
