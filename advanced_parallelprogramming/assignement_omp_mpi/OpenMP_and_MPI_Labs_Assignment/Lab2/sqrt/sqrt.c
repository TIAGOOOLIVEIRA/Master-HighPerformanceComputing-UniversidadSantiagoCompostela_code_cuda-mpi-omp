// Author: Wes Kendall
// Copyright 2012 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Program that computes the average of an array of elements in parallel using
// MPI_Scatter and MPI_Gather
//
// Modified to compute sqrt
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    //rand_nums[i] = (rand() * 100 / (float)RAND_MAX);
    rand_nums[i] = 9.0;
  }
  return rand_nums;
}

// Computes the sqrt for all elements an array of numbers
void compute_sqrt(float *array, int inicio, int fin) {
  int i;
  for (i = inicio; i < fin; i++) {
    array[i] = sqrt(array[i]);
  }
}

// test
float my_test(float *array,int num_elements) {
  float sum =0; 
  int i;
  for (i = 0; i < num_elements; i++) {
    sum += array[i];
  }
  return sum;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }
  
  double iniciow, finw;

  int num_elements_per_proc = atoi(argv[1]);
  // Seed the random number generator to get different results each time
    //srand(time(NULL));
  //1 para determinista
  srand(time(NULL));

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Create a random array of elements on the root process. Its total
  // size will be the number of elements per process times the number
  // of processes
  float *rand_nums = NULL;
  if (world_rank == 0) {
    rand_nums = create_rand_nums(num_elements_per_proc * world_size);
    printf("numbers created\n");
  }

  // For each process, create a buffer that will hold a subset of the entire
  // array
  float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
  assert(sub_rand_nums != NULL);

  // Scatter the random numbers from the root process to all processes in
  // the MPI world
  MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,
              num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

  printf("elements scattered proc:%d\n",world_rank);
  
  iniciow = MPI_Wtime();
  // Compute the sqrts of your subset
  compute_sqrt(sub_rand_nums, 0,num_elements_per_proc);

  printf("elements computed proc:%d\n",world_rank);
  
  // Gather all partial sqrts down to the root process
  float *result = NULL;
  if (world_rank == 0) {
    result = (float *)malloc(sizeof(float) * world_size * num_elements_per_proc);
    assert(result != NULL);
  }
  MPI_Gather(sub_rand_nums, num_elements_per_proc, MPI_FLOAT, result, num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

  printf("elements gathered proc:%d\n",world_rank);
  
  finw = MPI_Wtime();
  // 
  if (world_rank == 0) {
    printf("Time %g\n", finw-iniciow);
    printf("Test %g\n", my_test(result,world_size * num_elements_per_proc));
  }

  // Clean up
  if (world_rank == 0) {
    free(rand_nums);
    free(result);
  }
  free(sub_rand_nums);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
