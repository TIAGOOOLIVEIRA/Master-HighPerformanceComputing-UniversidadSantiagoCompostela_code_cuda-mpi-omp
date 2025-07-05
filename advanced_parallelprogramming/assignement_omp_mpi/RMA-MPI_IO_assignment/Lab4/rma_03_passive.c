/*
 * Exercise of one-sided operations with passive target synchronization
 *
 * Each process generates a list of random values and computes
 * a set of basic statistical metrics:
 * minimum, maximum, mean, and standard deviation
 *
 * Compile: mpicc -Wall -o rma_03_passive rma_03_passive.c -lm
 * Run: no arguments required
 * *  mpirun -np 4 ./rma_03_passive
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define A_SIZE 1000000000
#define V_MAX 100.0

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

typedef struct 
{
  double min;
  double max;
  double average;
  double standard_deviation;
} stats_t;

int main(int argc, char ** argv)
{
  int mpi_size, mpi_rank;
  double *my_array;
  stats_t stats;

  MEC(MPI_Init(&argc, &argv));
  MEC(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
  MEC(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  int block_size = A_SIZE / mpi_size;

  /* generate a set of random values */
  my_array = (double *) malloc(block_size * sizeof(double));
  srand(12345 + mpi_rank);
  for (int i=0; i<block_size; i++)
    my_array[i] = (1.0 * rand() / INT_MAX) * V_MAX;


  stats.min = stats.max = my_array[0];
  stats.average = 0;
  double sumvalues = 0;
  double sumsquared = 0;

  for (int i=0; i<block_size; i++) {
    if (my_array[i] < stats.min) stats.min = my_array[i];
    if (my_array[i] > stats.max) stats.max = my_array[i];
    sumvalues += my_array[i];
    sumsquared += my_array[i] * my_array[i];
  }
  stats.average = sumvalues / block_size;
  stats.standard_deviation =
    sqrt(
      (sumsquared + 2*sumvalues*stats.average + stats.average*stats.average)
      / block_size);

  printf("Process %d stats => min: %.4lf, max: %.4lf, avg: %.4lf, stddev: %.4lf\n",
         mpi_rank, stats.min, stats.max, stats.average, stats.standard_deviation);

  //Create shared window where each rank allocates its own region
  stats_t *shared_stats;
  MPI_Win win;

  MEC(MPI_Win_allocate(sizeof(stats_t), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_stats, &win));

  //Each rank writes its stats locally
  *shared_stats = stats;

  MEC(MPI_Barrier(MPI_COMM_WORLD));

  if (mpi_rank == 0) {
    stats_t global_stats = {.min = V_MAX, .max = 0, .average = 0, .standard_deviation = 0};
    stats_t *all_stats = malloc(mpi_size * sizeof(stats_t));

    for (int i = 0; i < mpi_size; i++) {
      MEC(MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win));
      /*MPI_Get:
        Data is transferred from the target memory to the origin process
        To complete the transfer a synchronization call must be made on the window involved
        The local buffer should not be accessed until the synchronization call is completed
        To read the content of the remote window (snd_buf) into the local variable rcv_buf
        Processes expose memory via windows; root process pulls remote data
        Decoupled communication: target processes don't need to call receive/send
        Requires explicit locking (MPI_Win_lock/unlock)
      */
      MEC(MPI_Get(&all_stats[i], sizeof(stats_t), MPI_BYTE, i, 0, sizeof(stats_t), MPI_BYTE, win));
      MEC(MPI_Win_unlock(i, win));
    }

    printf("  proc   minimum  maximum    mean  standard_dev\n");
    printf("-----------------------------------------------\n");
    for (int i = 0; i < mpi_size; i++) {
      printf("  %3d   %8.4lf %8.4lf %8.4lf %10.4lf\n",
             i,
             all_stats[i].min,
             all_stats[i].max,
             all_stats[i].average,
             all_stats[i].standard_deviation);

      if (all_stats[i].min < global_stats.min) global_stats.min = all_stats[i].min;
      if (all_stats[i].max > global_stats.max) global_stats.max = all_stats[i].max;
      global_stats.average += all_stats[i].average;
      global_stats.standard_deviation += all_stats[i].standard_deviation;
    }
    global_stats.average /= mpi_size;
    global_stats.standard_deviation /= mpi_size;
    printf("-----------------------------------------------\n");
    printf("        %8.4lf %8.4lf %8.4lf %10.4lf\n",
           global_stats.min,
           global_stats.max,
           global_stats.average,
           global_stats.standard_deviation);
    free(all_stats);
  }

  free(my_array);
  MEC(MPI_Win_free(&win));
  MEC(MPI_Finalize());
  return 0;
}
