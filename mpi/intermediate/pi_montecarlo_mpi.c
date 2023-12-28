#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

/* Constants for CHUNKSIZE and message tags */
#define CHUNKSIZE1 1000
#define CHUNKSIZE2 2000
#define REQUEST    1
#define REPLY      2

int main(int argc, char *argv[]) {
    int iter, in, out, i, iters, max, ix, iy, ranks[1], done, temp;
    double x, y, Pi, error, epsilon;
    int numprocs, myid, server, totalin, totalout, workerid;
    int rands[CHUNKSIZE1 > CHUNKSIZE2 ? CHUNKSIZE1 : CHUNKSIZE2], request;
    MPI_Comm world, workers1, workers2;  // Two separate worker communicators
    MPI_Group world_group, worker_group;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
    server = numprocs - 1;  // Last proc is the server

    // Determine whether the process belongs to workers1 or workers2
    MPI_Comm workers_comm;
    if (myid % 2 == 1) {  // Odd ranks belong to workers1
        MPI_Comm_split(world, 1, myid, &workers_comm);
    } else {  // Even ranks belong to workers2
        MPI_Comm_split(world, 2, myid, &workers_comm);
    }

    if (myid == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s epsilon\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        sscanf(argv[1], "%lf", &epsilon);
    }

    // Broadcast epsilon to all processes in workers_comm
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, workers_comm);

    MPI_Comm_group(world, &world_group);
    ranks[0] = server;
    MPI_Group_excl(world_group, 1, ranks, &worker_group);

    // Create separate communicator for workers1 and workers2
    if (myid % 2 == 1) {  // Odd ranks belong to workers1
        MPI_Comm_create(world, worker_group, &workers1);
    } else {  // Even ranks belong to workers2
        MPI_Comm_create(world, worker_group, &workers2);
    }

    MPI_Group_free(&worker_group);

    if (myid == server) {  // I am the rand server
        do {
            MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST,
                     world, &status);
            if (request) {
                for (i = 0; i < (myid % 2 == 1 ? CHUNKSIZE1 : CHUNKSIZE2); ) {
                    rands[i] = random();
                    if (rands[i] <= INT_MAX) i++;
                }
                MPI_Send(rands, i, MPI_INT,
                         status.MPI_SOURCE, REPLY, world);
            }
        } while (request > 0);
    } else {  // I am a worker process
        request = 1;
        done = in = out = 0;
        max = INT_MAX;  /* max int, for normalization */

        // Send initial request to the server
        MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
        MPI_Comm_rank(myid % 2 == 1 ? workers1 : workers2, &workerid);

        iter = 0;
        while (!done) {
            iter++;
            request = 1;
            MPI_Recv(rands, myid % 2 == 1 ? CHUNKSIZE1 : CHUNKSIZE2, MPI_INT, server, REPLY,
                     world, MPI_STATUS_IGNORE);
            for (i = 0; i < (myid % 2 == 1 ? CHUNKSIZE1 : CHUNKSIZE2); ) {
                x = (((double)rands[i++]) / max) * 2 - 1;
                y = (((double)rands[i++]) / max) * 2 - 1;
                if (x * x + y * y < 1.0)
                    in++;
                else
                    out++;
            }

            // Calculate local Pi
            Pi = (4.0 * in) / (in + out);
            error = fabs(Pi - 3.141592653589793238462643);
            done = (error < epsilon || (in + out) > 100000000);
            request = (done) ? 0 : 1;

            // Send request to the server or exit
            if (myid == 0) {
                printf("\rpi = %23.20f", Pi);
                MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            } else {
                if (request)
                    MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            }
        }
        if (myid % 2 == 1) {  // Odd ranks (workers1) calculate local Pi for workers1
            printf("\nLocal Pi for workers1: %23.20f\n", Pi);
        } else {  // Even ranks (workers2) calculate local Pi for workers2
            printf("\nLocal Pi for workers2: %23.20f\n", Pi);
        }

        MPI_Comm_free(myid % 2 == 1 ? &workers1 : &workers2);
    }

    if (myid == 0) {
        printf("\npoints: %d\nin: %d, out: %d, <ret> to exit\n",
               totalin + totalout, totalin, totalout);
        getchar();
    }
    MPI_Finalize();
    return 0;
}
