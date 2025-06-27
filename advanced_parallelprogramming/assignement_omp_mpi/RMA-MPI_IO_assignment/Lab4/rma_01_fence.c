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
 * Compile: mpicc -Wall -O3 -std=c99 -o 01_rma_fence 01_rma_fence.c
 * Run: no arguments required
 */

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 
 #define A_SIZE 1000
 
 int main(int argc, char ** argv)
 {
   int *a, b[2];
   int size, rank;
 
   MPI_Init(&argc, &argv);
 
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
   /* create private memory */
   //TODO: Memory should be allocated using MPI_Alloc_mem instead,
   //      to get the correct alignment
   a = (int *) malloc(A_SIZE * sizeof(int));
 
   a[0]=a[1]=rank; /* use private memory as usual */
 
   //TODO: Collectively declare memory in "a" as remotely accessible
   
   //TODO: Enclose RMA operations with calls to MPI_Win_fence
   
   int source = (rank + size - 1) % size;
   int target = (rank + 1) % size;
   b[0] = (rank+1) * 10;
   b[1] = (rank+1) * 20;
 
   //TODO: Replace these Two-Sided with One-Sided operations
 
   MPI_Request req;
   MPI_Isend(b, 2, MPI_INT,
            target, 0, MPI_COMM_WORLD, &req);
   MPI_Recv(a, 2, MPI_INT,
            source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 
   //TODO: Free MPI Window
   //TODO: Use MPI_Free_mem instead of free for memory allocated with MPI_Alloc_mem
   free(a);
   
   MPI_Finalize();
   return 0;
 }
 