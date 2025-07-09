#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ind(i,j) ((j)*(local_n+2)+(i))
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

void printarr(double *a, int n) {
    FILE *fp = fopen("heat.svg", "w");
    const int size = 5;
    fprintf(fp, "<html>\n<body>\n<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">");
    fprintf(fp, "\n<rect x=\"0\" y=\"0\" width=\"%i\" height=\"%i\" style=\"stroke-width:1;fill:rgb(0,0,0);stroke:rgb(0,0,0)\"/>", size*n, size*n);
    for(int i=1; i<n+1; ++i)
        for(int j=1; j<n+1; ++j) {
            int rgb = (a[(j)*(n+2)+i] > 0) ? (int)round(255.0*a[(j)*(n+2)+i]) : 0.0;
            if(rgb>255) rgb=255;
            if(rgb) fprintf(fp, "\n<rect x=\"%i\" y=\"%i\" width=\"%i\" height=\"%i\" style=\"stroke-width:1;fill:rgb(%i,0,0);stroke:rgb(%i,0,0)\"/>", size*(i-1), size*(j-1), size, size, rgb, rgb);
        }
    fprintf(fp, "</svg>\n</body>\n</html>");
    fclose(fp);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MEC(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MEC(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s grid_size energy niters\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    int energy = atoi(argv[2]);
    int niters = atoi(argv[3]);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MEC(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm));

    int coords[2];
    MEC(MPI_Cart_coords(cart_comm, rank, 2, coords));
    int up, down, left, right;
    MEC(MPI_Cart_shift(cart_comm, 0, 1, &left, &right));
    MEC(MPI_Cart_shift(cart_comm, 1, 1, &up, &down));

    int local_n = n / dims[0];
    int local_m = n / dims[1];

    double *aold = (double*)calloc((local_n+2)*(local_m+2), sizeof(double));
    double *anew = (double*)calloc((local_n+2)*(local_m+2), sizeof(double));
    double *tmp;

    #define nsources 3
    int sources[nsources][2] = {{n/2,n/2}, {n/3,n/3}, {n*4/5,n*8/9}};

    MPI_Datatype column_type;
    MEC(MPI_Type_vector(local_m, 1, local_n+2, MPI_DOUBLE, &column_type));
    MEC(MPI_Type_commit(&column_type));

    double t = -MPI_Wtime();
    for (int iter = 0; iter < niters; ++iter) {
        //halos
        MEC(MPI_Sendrecv(&aold[ind(1,1)], local_n, MPI_DOUBLE, up, 0, &aold[ind(1,0)], local_n, MPI_DOUBLE, down, 0, cart_comm, MPI_STATUS_IGNORE));
        MEC(MPI_Sendrecv(&aold[ind(1,local_m)], local_n, MPI_DOUBLE, down, 0, &aold[ind(1,local_m+1)], local_n, MPI_DOUBLE, up, 0, cart_comm, MPI_STATUS_IGNORE));
        MEC(MPI_Sendrecv(&aold[ind(1,1)], 1, column_type, left, 0, &aold[ind(0,1)], 1, column_type, right, 0, cart_comm, MPI_STATUS_IGNORE));
        MEC(MPI_Sendrecv(&aold[ind(local_n,1)], 1, column_type, right, 0, &aold[ind(local_n+1,1)], 1, column_type, left, 0, cart_comm, MPI_STATUS_IGNORE));

        double heat = 0.0;
        for (int j = 1; j <= local_m; ++j) {
            for (int i = 1; i <= local_n; ++i) {
                anew[ind(i,j)] = aold[ind(i,j)]/2.0 + (aold[ind(i-1,j)] + aold[ind(i+1,j)] + aold[ind(i,j-1)] + aold[ind(i,j+1)])/4.0/2.0;
                heat += anew[ind(i,j)];
            }
        }
        tmp = anew; anew = aold; aold = tmp;
    }
    t += MPI_Wtime();

    double total_heat;
    MEC(MPI_Reduce(&t, &total_heat, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm));
    if (rank == 0){
        printf("Time: %f\n", total_heat);
        //printfarr(aold, n);
        //printarr(anew, n);
    } 

    free(aold);
    free(anew);
    MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}
