/*
 *Intermediate MPI Lab - Parallel Programming
 Assingment: Intermediate MPI (Exercises 4, 8, 11 and 13)
 Student: Tiago de Souza Oliveira

 Exercise 11 - LAB 11: Reduce_scatter
* Explain how the function MPI_Reduce_scatter works in this example.
    Essentially, the MPI_Reduce_scatter function performs a combination of reduction and scatter operations.
    In this case, a reduction operation (MPI_SUM) is applied across all processes to combine corresponding elements from the sendbuf arrays of each process.
    Then each process receives a portion of the reduced array into its recvbuf. After the reduction, the results are distributed (scattered) to the processes according to the recvcounts array.


To compile
    module load intel impi
    mpicc -o exercise11_Reduce_scatter exercise11_Reduce_scatter.c

To submit and wath the job:
  sbatch run_exercise11_Reduce_scatter.sh
  watch squeue -u curso370 
 
This job is executed from a shell script "run_exercise11_Reduce_scatter.sh" composed by the following instructions:

#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J run_exercise11_Reduce_scatter       # Job name
#SBATCH -o run_exercise11_Reduce_scatter.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e run_exercise11_Reduce_scatter.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:10:00 #(10 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=4

module load intel impi

mpirun -np $SLURM_NTASKS ./exercise11_Reduce_scatter

*/


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
 
int main(int argc, char *argv[])
{
int err = 0;
int *sendbuf, recvbuf, *recvcounts;
int npes, myid, i, sumval;
 
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &npes);	
MPI_Comm_rank(MPI_COMM_WORLD, &myid);

sendbuf = (int *) malloc( npes * sizeof(int) );

for (i=0; i<npes; i++) 
        sendbuf[i] = myid + i;

recvcounts = (int *)malloc( npes * sizeof(int) );
for (i=0; i<npes; i++) 
       recvcounts[i] = 1;
 
MPI_Reduce_scatter(sendbuf, &recvbuf, recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
 
sumval = npes * myid + ((npes - 1) * npes)/2;

printf("\n P(%d) got %d expected %d\n", myid, recvbuf, sumval );fflush(stdout);

MPI_Finalize( );

}
