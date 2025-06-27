/*
 * Exercise of one-sided operations with passive target synchronization
 * 
 * Each process generates a list of random values and computes
 * a set of basic statistical metrics:
 * minimum, maximum, mean, and standard deviation
 * 
 * Compile: mpicc -Wall -o rma_03_passive rma_03_passive.c -lm
 * Run: no arguments required
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define A_SIZE 1000000000
#define V_MAX 100.0

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

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int block_size = A_SIZE / mpi_size;

  /* generate a set of random values */
  my_array = (double *) malloc(block_size * sizeof(double));
  srand(12345 + mpi_rank);
  for (int i=0; i<block_size; i++)
    my_array[i] = (1.0 * rand() / INT_MAX) * V_MAX;

  /* initialize stats */
  stats.min = stats.max = my_array[0];
  stats.average = 0;
  double sumvalues = 0;
  double sumsquared = 0;

  /* compute stats */
  for (int i=0; i<block_size; i++)
  {
    if (my_array[i] < stats.min)
      stats.min = my_array[i];
    if (my_array[i] > stats.max)
      stats.max = my_array[i];
    sumvalues += my_array[i];
    sumsquared += my_array[i] * my_array[i];
  }
  stats.average = sumvalues / block_size;
  stats.standard_deviation =
    sqrt(
      (sumsquared + 2*sumvalues*stats.average + stats.average*stats.average)
      / block_size);

  //TODO: Replace the collective operations with One-Sided communications
  stats_t global_stats;

  MPI_Reduce(&stats.min, &global_stats.min, 1, MPI_DOUBLE,
             MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&stats.max, &global_stats.max, 1, MPI_DOUBLE,
             MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&stats.average, &global_stats.average, 1, MPI_DOUBLE,
             MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&stats.standard_deviation, &global_stats.standard_deviation, 1, MPI_DOUBLE,
             MPI_SUM, 0, MPI_COMM_WORLD);
  global_stats.average /= mpi_size;
  global_stats.standard_deviation /= mpi_size;

  if (!mpi_rank)
  {
    //TODO: Replace the results gathering with One-Sided communications
    //      Start by creating a collective Window, but only the root process
    //      should have memory attached to it.
    stats_t perproc_stats[mpi_size];
    MPI_Gather(&stats, sizeof(stats_t), MPI_BYTE,
               perproc_stats, sizeof(stats_t), MPI_BYTE,
               0, MPI_COMM_WORLD);

    /* print individual stats */
    printf("  proc   minimum  maximum    mean  standard_dev\n");
    printf("-----------------------------------------------\n");
    for (int i=0; i<mpi_size; i++)
    {
      printf("  %3d   %8.4lf %8.4lf %8.4lf %10.4lf\n",
             i,
             perproc_stats[i].min,
             perproc_stats[i].max,
             perproc_stats[i].average,
             perproc_stats[i].standard_deviation);
    }
    printf("-----------------------------------------------\n");
    printf("        %8.4lf %8.4lf %8.4lf %10.4lf\n",
             global_stats.min,
             global_stats.max,
             global_stats.average,
             global_stats.standard_deviation);
  }
  else
  {
    MPI_Gather(&stats, sizeof(stats_t), MPI_BYTE,
               0, sizeof(stats_t), MPI_BYTE,
               0, MPI_COMM_WORLD);
  }
  
  free(my_array);
    
  MPI_Finalize();
  return 0;
}

