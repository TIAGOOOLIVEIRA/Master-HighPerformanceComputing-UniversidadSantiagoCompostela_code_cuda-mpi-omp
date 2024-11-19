/*
 *Intermediate MPI Lab - Parallel Programming
 Assingment: Intermediate MPI (Exercises 4, 8, 11 and 13)
 Student: Tiago de Souza Oliveira

 Exercise 8 - LAB 8: Scatter/Gather

To compile
    module load intel impi
    mpicc -o exercise8_gather exercise8_gather.c

To submit and wath the job:
  sbatch run_exercise8_gather.sh
  watch squeue -u curso370 
 
This job is executed from a shell script "run_exercise8_gather.sh" composed by the following instructions:

#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J exercise8_gather       # Job name
#SBATCH -o exercise8_gather.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e exercise8_gather.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:10:00 #(10 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=2

module load intel impi

mpirun -np $SLURM_NTASKS ./exercise8_gather

*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int myid, npes, i;
    int *gathered_ids;
    int my_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);    
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    my_rank = myid;

    if (myid == 0) {
        gathered_ids = (int *)malloc(npes * sizeof(int));
    } else {
        gathered_ids = NULL;
    }

    MPI_Gather(&my_rank, 1, MPI_INT, gathered_ids, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("\nProcess #0 gather ranks: ");
        for (i = 0; i < npes; i++) {
            printf("\n#%d ", gathered_ids[i]);
        }
        printf("\n");
        free(gathered_ids);
    }

    MPI_Finalize();
    return 0;
}
