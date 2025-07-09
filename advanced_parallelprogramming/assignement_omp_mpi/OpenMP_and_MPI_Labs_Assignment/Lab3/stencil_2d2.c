include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ind(i, j) ((j) * (bx + 2) + (i))  //idx in local (bx+2)x(by+2) array with halos
#define MEC(call) { \
    int res = (call); \
    if (res != MPI_SUCCESS) { \
        char errstr[MPI_MAX_ERROR_STRING]; \
        int errlen; \
        MPI_Error_string(res, errstr, &errlen); \
        fprintf(stderr, "MPI error calling %s at line %d: %s\n", #call, __LINE__, errstr); \
        MPI_Abort(MPI_COMM_WORLD, res); \
    } \
}

extern void printarr_par(int iter, double* array, int size, int px, int py,
                         int rx, int ry, int bx, int by, int offx, int offy, MPI_Comm comm);

int main(int argc, char **argv) {
    MEC(MPI_Init(&argc, &argv));

    int rank, size;
    MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (argc != 4) {
        if (rank == 0) fprintf(stderr, "Usage: %s n energy niters\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n = atoi(argv[1]);       // Global grid size (n x n)
    int energy = atoi(argv[2]);  // Heat injection
    int niters = atoi(argv[3]);  // Iterations

    // Cartesian topology
    int dims[2] = {0, 0};
    MEC(MPI_Dims_create(size, 2, dims));
    int px = dims[0], py = dims[1];
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MEC(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm));

    int coords[2];
    MEC(MPI_Cart_coords(cart_comm, rank, 2, coords));
    int rx = coords[0], ry = coords[1];

    //Subdomain dimensions
    int bx = n / px, by = n / py;
    int offx = rx * bx, offy = ry * by;

    //Allocate 2D arrays with 1-cell halo
    double *aold = calloc((bx + 2) * (by + 2), sizeof(double));
    double *anew = calloc((bx + 2) * (by + 2), sizeof(double));
    double *tmp;

    //heat sources (in global coordinates)
    #define nsources 3
    int sources[nsources][2] = {{n/2,n/2}, {n/3,n/3}, {n*4/5,n*8/9}};

    //Neighbor ranks
    int north, south, west, east;
    MEC(MPI_Cart_shift(cart_comm, 1, 1, &north, &south));
    MEC(MPI_Cart_shift(cart_comm, 0, 1, &west, &east));

    //Prepare send/receive buffers for halo exchange
    double *send_north = &aold[ind(1,1)];
    double *recv_north = &aold[ind(1,0)];
    double *send_south = &aold[ind(1,by)];
    double *recv_south = &aold[ind(1,by+1)];

    double *send_west = malloc(by * sizeof(double));
    double *recv_west = malloc(by * sizeof(double));
    double *send_east = malloc(by * sizeof(double));
    double *recv_east = malloc(by * sizeof(double));

    double t0 = MPI_Wtime();
    double heat = 0.0;

    for (int iter = 0; iter < niters; ++iter) {
        //vertical data
        for (int j = 0; j < by; j++) {
            send_west[j] = aold[ind(1, j+1)];
            send_east[j] = aold[ind(bx, j+1)];
        }

        //Halo exchange: North/South (rows)
        MEC(MPI_Sendrecv(send_north, bx, MPI_DOUBLE, north, 0,
                         recv_south, bx, MPI_DOUBLE, south, 0, cart_comm, MPI_STATUS_IGNORE));
        MEC(MPI_Sendrecv(send_south, bx, MPI_DOUBLE, south, 1,
                         recv_north, bx, MPI_DOUBLE, north, 1, cart_comm, MPI_STATUS_IGNORE));

        //Halo exchange: West/East (columns)
        MEC(MPI_Sendrecv(send_west, by, MPI_DOUBLE, west, 2,
                         recv_east, by, MPI_DOUBLE, east, 2, cart_comm, MPI_STATUS_IGNORE));
        MEC(MPI_Sendrecv(send_east, by, MPI_DOUBLE, east, 3,
                         recv_west, by, MPI_DOUBLE, west, 3, cart_comm, MPI_STATUS_IGNORE));

        //Unpack vertical halos
        for (int j = 0; j < by; j++) {
            aold[ind(0, j+1)] = recv_west[j];
            aold[ind(bx+1, j+1)] = recv_east[j];
        }

        //Compute stencil
        heat = 0.0;
        for (int j = 1; j <= by; j++) {
            for (int i = 1; i <= bx; i++) {
                anew[ind(i,j)] = aold[ind(i,j)] / 2.0 +
                    (aold[ind(i-1,j)] + aold[ind(i+1,j)] +
                     aold[ind(i,j-1)] + aold[ind(i,j+1)]) / 8.0;
                heat += anew[ind(i,j)];
            }
        }

        //Inject heat sources
        for (int s = 0; s < nsources; s++) {
            int gi = sources[s][0], gj = sources[s][1];
            if (gi >= offx && gi < offx+bx && gj >= offy && gj < offy+by) {
                anew[ind(gi-offx+1, gj-offy+1)] += energy;
            }
        }

        tmp = anew; anew = aold; aold = tmp;
    }

    double t1 = MPI_Wtime();
    if (rank == 0)
        printf("Time: %f\n", t1 - t0);

    //Output BMP image
    printarr_par(niters, aold, n, px, py, rx, ry, bx, by, offx, offy, cart_comm);

    free(aold); free(anew);
    free(send_west); free(send_east); free(recv_west); free(recv_east);
    MEC(MPI_Finalize());
    return 0;
}