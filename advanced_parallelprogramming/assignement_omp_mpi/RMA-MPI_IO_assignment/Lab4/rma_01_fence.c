/*
 * Example of active target synchronization using Fences
 *
 * Each process declares 2 arrays:
 *   "int * a" dynamically allocated on the heap
 *   "int b[2]" statically allocated on the stack
 * "a" positions 1 and 2 are arbitrarily initialized to "rank"
 * "b" is arbitrarily initialized to {(rank+1)*10, (rank+1)*20}
 *
 * Each process will use RMA to put its "b" values into "a" array in the
 * private memory of the next process following a ring topology
 *
 * 0 -> 1 -> 2 -> .. -> N -> 0
 * 
 * Compile: mpicc -Wall -O3 -std=c99 -o rma_01_fence rma_01_fence.c
 * Run: no arguments required
 */

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 
 #define A_SIZE 1000
 
 #define MEC(call) { int res = call; \
   if (res != MPI_SUCCESS) { \
     char error_string[MPI_MAX_ERROR_STRING]; \
     int length; \
     MPI_Error_string(res, error_string, &length); \
     fprintf(stderr, "MPI Error: %s\nCall: %s\n", error_string, #call); \
     MPI_Abort(MPI_COMM_WORLD, res); \
   } \
 }
 
 int main(int argc, char **argv)
 {
   int *a, b[2];
   int size, rank;
 
   MEC(MPI_Init(&argc, &argv));
   MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));
   MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
 
   //MPI_Alloc_mem for proper alignment and performance
   MEC(MPI_Alloc_mem(A_SIZE * sizeof(int), MPI_INFO_NULL, &a));
 
   a[0] = a[1] = rank;
 
   //memory in "a" as remotely accessible using MPI_Win_create
   MPI_Win win;
   MEC(MPI_Win_create(a, A_SIZE * sizeof(int), sizeof(int), MPI_INFO_NULL,
                      MPI_COMM_WORLD, &win));
 
   int source = (rank + size - 1) % size;
   int target = (rank + 1) % size;
   b[0] = (rank + 1) * 10;
   b[1] = (rank + 1) * 20;
 
   //Fence to start RMA
   MEC(MPI_Win_fence(0, win));
 
   //One-sided op replacing two-sided Isend/Recv
   //puts the two values of b into the start of the next rank's array
   MEC(MPI_Put(b, 2, MPI_INT, target, 0, 2, MPI_INT, win));
 
   //Fence to end RMA epoch (ensures put is complete) - like a barrier
   //a synchronization point for all processes
   MEC(MPI_Win_fence(0, win));
 
   //Print the values received in a
   //a is private memory of the process, but we can access it after the fence
   printf("Rank %d received values in a: %d %d (from rank %d)\n", rank, a[0], a[1], source);
 
   //Free MPI Window and memory
   MEC(MPI_Win_free(&win));
   MEC(MPI_Free_mem(a));
 
   MEC(MPI_Finalize());
   return 0;
 }
 