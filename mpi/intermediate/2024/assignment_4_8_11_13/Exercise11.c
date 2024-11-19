/*
 *Intermediate MPI Lab - Parallel Programming
 Assingment: Intermediate MPI (Exercises 4, 8, 11 and 13)
 Student: Tiago de Souza Oliveira

 Exercise 11 - smoothing grayscale image using 2D Cartesian communicator
 
Notes:
    By leveraging MPI_Cart_shift, the program efficiently distributes work across processes and manages inter-process communication for edge cases during the smoothing operation

To compile
    module load intel impi
    mpicc -fopenmp -o smoothing_grayscale Exercise11.c -lm

To submit and wath the job:
  sbatch Exercise11.sh
  watch squeue -u curso370 

To visualize the image:
    #module load imagemagick
    display smoothed_image.pgm&


This job is executed from a shell script "Exercise11.sh" composed by the following instructions:

#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J run_smoothing_grayscale       # Job name
#SBATCH -o run_smoothing_grayscale.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e run_smoothing_grayscale.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -c 32 #(64 cores per job)
#SBATCH -t 00:25:00 #(10 min of execution time) 
#SBATCH --mem=16GB #(4GB of memory) 

#SBATCH --ntasks=9

module load intel impi

OMP_NUM_THREADS=8 mpirun -np $SLURM_NTASKS ./smoothing_grayscale

*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <omp.h>

#define IMAGE_SIZE 12
#define PIXEL_MAX 255

void initialize_image(int image[IMAGE_SIZE][IMAGE_SIZE]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            image[i][j] = rand() % (PIXEL_MAX + 1);
        }
    }
}

void print_image(const char *label, int image[IMAGE_SIZE][IMAGE_SIZE]) {
    printf("\n%s:\n", label);
    #pragma omp parallel for
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            printf("%3d ", image[i][j]);
        }
        printf("\n");
    }
}

//Function to save the grayscale image as a PGM file
void save_to_pgm(const char *filename, int image[IMAGE_SIZE][IMAGE_SIZE]) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }

    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", IMAGE_SIZE, IMAGE_SIZE);
    fprintf(file, "%d\n", PIXEL_MAX);

    #pragma omp parallel for
    for (int i = 0; i < IMAGE_SIZE; i++) {
        char row_buffer[IMAGE_SIZE * 5]; //Buffer for one row (assuming 4 digits + space per pixel)
        int pos = 0;
        for (int j = 0; j < IMAGE_SIZE; j++) {
            pos += sprintf(&row_buffer[pos], "%d ", image[i][j]);
        }
        row_buffer[pos - 1] = '\n';

        #pragma omp critical
        {
            fprintf(file, "%s", row_buffer);
        }
    }

    fclose(file);
    printf("PGM image saved to %s\n", filename);
}


int main(int argc, char *argv[]) {
    int rank, size, dims[2], coords[2], periods[2] = {0, 0}, reorder = 1;
    int left, right, up, down;
    int image[IMAGE_SIZE][IMAGE_SIZE];
    int smoothed_image[IMAGE_SIZE][IMAGE_SIZE];
    MPI_Comm cart_comm;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0)
            fprintf(stderr, "Total processes must be square!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    dims[0] = dims[1] = grid_size;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Determine neighbors in the Cartesian grid
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    if (rank == 0) {
        initialize_image(image);
        print_image("Original Image", image);
    }

    int submatrix_size = IMAGE_SIZE / grid_size;
    int submatrix[submatrix_size][submatrix_size];
    int smoothed_submatrix[submatrix_size][submatrix_size];

    //Scatter the image to all processes
    MPI_Datatype submatrix_type;
    MPI_Type_vector(submatrix_size, submatrix_size, IMAGE_SIZE, MPI_INT, &submatrix_type);
    MPI_Type_create_resized(submatrix_type, 0, sizeof(int), &submatrix_type);
    MPI_Type_commit(&submatrix_type);

    int sendcounts[size];
    int displs[size];
    if (rank == 0) {
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                sendcounts[i * grid_size + j] = 1;
                displs[i * grid_size + j] = i * IMAGE_SIZE * submatrix_size + j * submatrix_size;
            }
        }
    }

    MPI_Scatterv(image, sendcounts, displs, submatrix_type, submatrix, submatrix_size * submatrix_size, MPI_INT, 0, cart_comm);

    for (int i = 0; i < submatrix_size; i++) {
        for (int j = 0; j < submatrix_size; j++) {
            int count = 2;
            int sum = 2 * submatrix[i][j];

            if (i > 0) { sum += submatrix[i - 1][j]; count++; }
            if (i < submatrix_size - 1) { sum += submatrix[i + 1][j]; count++; }
            if (j > 0) { sum += submatrix[i][j - 1]; count++; }
            if (j < submatrix_size - 1) { sum += submatrix[i][j + 1]; count++; }

            smoothed_submatrix[i][j] = sum / count;
        }
    }

    //Smoothed submatrices back to the root process
    MPI_Gatherv(smoothed_submatrix, submatrix_size * submatrix_size, MPI_INT,
                smoothed_image, sendcounts, displs, submatrix_type, 0, cart_comm);

    if (rank == 0) {
        print_image("Smoothed Image", smoothed_image);
        save_to_pgm("smoothed_image.pgm", smoothed_image);
    }

    MPI_Type_free(&submatrix_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
