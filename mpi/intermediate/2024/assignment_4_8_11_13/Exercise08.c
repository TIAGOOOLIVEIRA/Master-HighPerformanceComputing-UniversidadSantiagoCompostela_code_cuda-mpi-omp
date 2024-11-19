/*
 *Intermediate MPI Lab - Parallel Programming
 Assingment: Intermediate MPI (Exercises 4, 8, 11 and 13)
 Student: Tiago de Souza Oliveira

 Exercise 8 - Modify the following MPI application written in language C in order to create two communicators workers1 and workers2...
 
To compile
    module load intel impi
    mpicc -o pimontecarlo Exercise08.c -lm


To submit and wath the job:
  sbatch Exercise08.sh
  watch squeue -u curso370 


This job is executed from a shell script "Exercise08.sh" composed by the following instructions:

#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J pimontecarlo       # Job name
#SBATCH -o pimontecarlo.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e pimontecarlo.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:10:00 #(10 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=2

module load intel impi

mpirun -np $SLURM_NTASKS ./pimontecarlo 0.0001

*/


#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#define CHUNKSIZEWORKER1 1000
#define CHUNKSIZEWORKER2 2000

/* message tags */
#define REQUEST  1
#define REPLY    2

int main(int argc, char *argv[])
{
    int iter, in, out, i, max, ix, iy, ranks[1], done, temp;
    double x, y, Pi1, Pi2, globalPi, error, epsilon;
    int numprocs, myid, server, totalin, totalout, workerid;
    int *rands, request, chunk_size;
    MPI_Comm world, workers1, workers2;
    MPI_Group world_group, workers1_group, workers2_group;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
    
	server = numprocs - 1;

    if (myid == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s epsilon\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        sscanf(argv[1], "%lf", &epsilon);
    }
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*Create worker groups */
    MPI_Comm_group(world, &world_group);

    // Even ranks go to workers2, odd ranks go to workers1
    int workers1_ranks[(numprocs - 1) / 2], workers2_ranks[(numprocs - 1) / 2];
    int w1_idx = 0, w2_idx = 0;
    for (int r = 0; r < numprocs - 1; r++) {
        if (r % 2 == 0)
            workers2_ranks[w2_idx++] = r;
        else
            workers1_ranks[w1_idx++] = r;
    }

    MPI_Group_incl(world_group, w1_idx, workers1_ranks, &workers1_group);
    MPI_Group_incl(world_group, w2_idx, workers2_ranks, &workers2_group);

    MPI_Comm_create(world, workers1_group, &workers1);
    MPI_Comm_create(world, workers2_group, &workers2);

    MPI_Group_free(&world_group);
    MPI_Group_free(&workers1_group);
    MPI_Group_free(&workers2_group);

    if (myid == server) {
        do {
            MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status);
            if (request) {
                int chunk_size = (status.MPI_SOURCE % 2 == 0) ? CHUNKSIZEWORKER2 : CHUNKSIZEWORKER1;
                rands = (int *)malloc(chunk_size * sizeof(int));
                for (i = 0; i < chunk_size;) {
                    rands[i] = random();
                    if (rands[i] <= INT_MAX) i++;
                }
                MPI_Send(rands, chunk_size, MPI_INT, status.MPI_SOURCE, REPLY, world);
                free(rands);
            }
        } while (request > 0);
    } else { /*worker process */
        request = 1;
        done = in = out = 0;
        max = INT_MAX;

        /*communicator and chunk size */
        MPI_Comm current_comm = (myid % 2 == 0) ? workers2 : workers1;
        chunk_size = (myid % 2 == 0) ? CHUNKSIZEWORKER2 : CHUNKSIZEWORKER1;

        rands = (int *)malloc(chunk_size * sizeof(int));

        while (!done) {
            MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            MPI_Recv(rands, chunk_size, MPI_INT, server, REPLY, world, MPI_STATUS_IGNORE);

            for (i = 0; i < chunk_size; i += 2) {
                x = (((double)rands[i]) / max) * 2 - 1;
                y = (((double)rands[i + 1]) / max) * 2 - 1;
                if (x * x + y * y < 1.0)
                    in++;
                else
                    out++;
            }

            int group_totalin, group_totalout;
            MPI_Allreduce(&in, &group_totalin, 1, MPI_INT, MPI_SUM, current_comm);
            MPI_Allreduce(&out, &group_totalout, 1, MPI_INT, MPI_SUM, current_comm);

            double group_pi = (4.0 * group_totalin) / (group_totalin + group_totalout);
            if (myid % 2 == 0)
                Pi2 = group_pi;
            else
                Pi1 = group_pi;

            totalin = in;
            totalout = out;

            MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, world);
            MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, world);

            globalPi = (4.0 * totalin) / (totalin + totalout);
            error = fabs(globalPi - 3.141592653589793238462643);

            done = (error < epsilon || (totalin + totalout) > 100000000);
            request = done ? 0 : 1;

            if (myid == 0) {
                printf("\rGlobal pi = %23.20f, Workers1 pi = %23.20f, Workers2 pi = %23.20f", globalPi, Pi1, Pi2);
            }
        }

        free(rands);
        if (current_comm != MPI_COMM_NULL)
            MPI_Comm_free(&current_comm);
    }

    MPI_Finalize();
    return 0;
}
