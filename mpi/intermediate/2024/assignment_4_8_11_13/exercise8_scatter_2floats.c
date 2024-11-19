/*
 *Intermediate MPI Lab - Parallel Programming
 Assingment: Intermediate MPI (Exercises 4, 8, 11 and 13)
 Student: Tiago de Souza Oliveira

 Exercise 8 - LAB 8: Scatter/Gather

To compile
    module load intel impi
    mpicc -o exercise8_scatter_2floats exercise8_scatter_2floats.c

To submit and wath the job:
  sbatch Exercise04.sh
  watch squeue -u curso370 
 
This job is executed from a shell script "run_exercise8_scatter_2floats.sh" composed by the following instructions:

#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J exercise8_scatter_2floats       # Job name
#SBATCH -o exercise8_scatter_2floats.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e exercise8_scatter_2floats.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:10:00 #(10 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=2

module load intel impi

mpirun -np $SLURM_NTASKS ./exercise8_scatter_2floats
 
*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int myid, npes, i;
    float *vector;
    float vector_rec[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);    
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        vector = (float *)malloc(2 * npes * sizeof(float));
        for (i = 0; i < 2 * npes; i++) {
            vector[i] = i * 1.1;
        }
    } else {
        vector = NULL;
    }

    MPI_Scatter(vector, 2, MPI_FLOAT, vector_rec, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    printf("I am process %d, received vector_rec: [%f, %f]\n", myid, vector_rec[0], vector_rec[1]);

    if (myid == 0) {
        free(vector);
    }

    MPI_Finalize();
    return 0;
}