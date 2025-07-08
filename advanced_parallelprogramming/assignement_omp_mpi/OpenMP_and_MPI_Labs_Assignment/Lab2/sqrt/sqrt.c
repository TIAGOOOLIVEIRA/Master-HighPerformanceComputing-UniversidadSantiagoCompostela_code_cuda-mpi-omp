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
#include <string.h>

#define MEC(call) { \
     int res = (call); \
     if (res != MPI_SUCCESS) { \
         char err_str[256]; int err_len; \
         MPI_Error_string(res, err_str, &err_len); \
         fprintf(stderr, "[Rank %d] MPI error at line %d: %s\n", rank, __LINE__, err_str); \
         MPI_Abort(MPI_COMM_WORLD, res); \
     } \
}

float *create_rand_nums(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  for (int i = 0; i < num_elements; i++) {
    rand_nums[i] = 9.0f;
  }
  return rand_nums;
}

void compute_sqrt(float *array, int inicio, int fin) {
  for (int i = inicio; i < fin; i++) {
    array[i] = sqrt(array[i]);
  }
}

float my_test(float *array, int num_elements) {
  float sum = 0.0f;
  for (int i = 0; i < num_elements; i++) {
    sum += array[i];
  }
  return sum;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s num_elements_per_proc num_pipeline_steps\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  double iniciow, finw;
  int num_elements_per_proc = atoi(argv[1]);
  int steps = atoi(argv[2]);

  MPI_Init(NULL, NULL);

  int rank, size;
  MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));

  float *rand_nums = NULL;
  if (rank == 0) {
    rand_nums = create_rand_nums(num_elements_per_proc * size);
  }

  float *sub_rand_nums = (float *)malloc(sizeof(float) * num_elements_per_proc);
  assert(sub_rand_nums != NULL);

  MEC(MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT,
                  sub_rand_nums, num_elements_per_proc, MPI_FLOAT,
                  0, MPI_COMM_WORLD));

  float *result = NULL;
  if (rank == 0) {
    result = (float *)malloc(sizeof(float) * size * num_elements_per_proc);
    assert(result != NULL);
  }

  float *step_buffer = (float *)malloc(sizeof(float) * num_elements_per_proc / steps);
  MPI_Request requests[steps];

  iniciow = MPI_Wtime();

  for (int s = 0; s < steps; s++) {
    int chunk = num_elements_per_proc / steps;
    int start = s * chunk;
    int end = (s == steps - 1) ? num_elements_per_proc : start + chunk;

    compute_sqrt(sub_rand_nums, start, end);
    memcpy(step_buffer, &sub_rand_nums[start], (end - start) * sizeof(float));

    MEC(MPI_Igather(step_buffer, end - start, MPI_FLOAT,
                    rank == 0 ? &result[start] : NULL, end - start, MPI_FLOAT,
                    0, MPI_COMM_WORLD, &requests[s]));
  }

  MEC(MPI_Waitall(steps, requests, MPI_STATUSES_IGNORE));

  finw = MPI_Wtime();

  if (rank == 0) {
    printf("Total execution time with %d steps: %g seconds\n", steps, finw - iniciow);
    printf("Test sum: %g\n", my_test(result, size * num_elements_per_proc));
    free(rand_nums);
    free(result);
  }

  free(sub_rand_nums);
  free(step_buffer);

  MEC(MPI_Barrier(MPI_COMM_WORLD));
  MEC(MPI_Finalize());
  return 0;
}