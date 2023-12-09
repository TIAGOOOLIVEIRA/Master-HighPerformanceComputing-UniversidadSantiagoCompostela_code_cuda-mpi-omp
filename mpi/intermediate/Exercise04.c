/*
 *Intermediate MPI Lab - Parallel Programming
 Student: Tiago de Souza Oliveira

 Exercise 4 - Use MPI Type indexed to create a derived datatype that corresponds to the lower triangular part of the following 4x4
 
 This job is executed from a shell script "Exercise04.sh" composed by the following instructions:
 
#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error
#SBATCH --time=0-00:05:00 #requested time to run the job
#SBATCH -n 2   # number of concurrent jobs (64 cores/node)
#SBATCH --mem-per-cpu 1G

srun ./lowertriang

 * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SENDER 0
#define RECEIVER 1

void gen_array(float arrayR[], const unsigned int n);
void extract_lowertriang(float arrayOriginal[], float arrayR[], const unsigned int row, const unsigned int col);
void extract_len_displacement(float arrayOriginal[], unsigned int arrayLen[], unsigned int arrayDisplacement[], const unsigned int row, const unsigned int col);


int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);

    int nprocesses;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocesses);

    if(nprocesses != 2)
    {
        printf("Application only for 2 processes total: 1 Sender -> 1 Receiver.\n");

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //defining the size of the quadratic  Matrix.
    //to have dimension definitions values across processors
    unsigned int row = 4;
    unsigned int col = 4;
    unsigned int n = row * col;
    
    if(SENDER == rank){
      unsigned int numBytes = n *  sizeof(float);
      float *vectorA = (float *) malloc(numBytes);
    
      unsigned int *vectorL = (unsigned int *) malloc(row * sizeof(unsigned int));
      unsigned int *vectorD = (unsigned int *) malloc(row * sizeof(unsigned int));

      //float *vectorB = (float *) malloc(numBytes);
      
      gen_array(vectorA, n);
      extract_len_displacement(vectorA, vectorL, vectorD, row, col);

      // Create the datatype
      MPI_Datatype triangle_type;
     
      //since this expects to receive a quadratic matrix, taken row or col values behaves as the same.
      //function extract_len_displacement generalizes the expected shape below for a 4 x 4 Matrix.
      //int lengths[4] = { 1, 2, 3 ,4};
      //int displacements[4] = { 0, 4, 8, 12};
      MPI_Type_indexed(row, vectorL, vectorD, MPI_INT, &triangle_type);
      MPI_Type_commit(&triangle_type);


      //to validate concept of lower triangle from a matrix 
      //extract_lowertriang(vectorA, vectorB, row, col);
      
      MPI_Request request;
      MPI_Send(vectorA, 1, triangle_type, RECEIVER, 0, MPI_COMM_WORLD);


      printf("Sender pid %d Lenght and Displacement \n", SENDER);
      for(unsigned int i = 0; i < row; i++)
       printf("Len per row %d Displacement: %d \n", vectorL[i], vectorD[i]);

      free(vectorA);      
      //free(vectorB);
      free(vectorL);
      free(vectorD);
    }
    else if(RECEIVER == rank){
      //10, for matrix size of 16 (4 x 4)
      //Represents the sum of quantity elements collected per row, where (col < ( row + 1))
      //value can be taken from the sum reduction from "vectorL", so it is hardcoded here for the sake of simplicity.
      unsigned int sizeElementsLowerTriang = 10;
      float received[sizeElementsLowerTriang];
      MPI_Recv(&received, sizeElementsLowerTriang, MPI_INT, SENDER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      
      printf("Receiver pid: %d received values:\n", RECEIVER);
      for(unsigned int i = 0; i < sizeElementsLowerTriang; i++)
       printf("value: %f position: %d \n", received[i], i);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}



void gen_array(float arrayR[], const unsigned int n){
      
    for(unsigned int i = 0; i < n; i++) 
       arrayR[i] = (float)i;
}

void extract_lowertriang(float arrayOriginal[], float arrayR[], const unsigned int row, const unsigned int col){
    //https://en.wikipedia.org/wiki/Triangular_matrix
    
    //printf("loops for 2d lower triang \n\n");
    for(unsigned int i = 0; i < row; i ++){
     for(unsigned int j = 0; j < col; j++){
       unsigned int idx2d = i * row + j; 
       float aux = arrayOriginal[idx2d];

       if(j < (i+1))
        arrayR[idx2d] = arrayOriginal[idx2d];
       else
	arrayR[idx2d] = -1;
      //printf("i: %d j: %d  idx2d: %d array[idx2d]: %f\n", i, j, idx2d, aux);
     }
    }
}

void extract_len_displacement(float arrayOriginal[], unsigned int arrayLen[], unsigned int arrayDisplacement[], const unsigned int row, const unsigned int col){
    //https://en.wikipedia.org/wiki/Triangular_matrix
    
    for(unsigned int i = 0; i < row; i ++){
     unsigned int lenperrow = 0;
     unsigned int idxdisplace = 0;
     for(unsigned int j = 0; j < col; j++){
       unsigned int idx2d = i * row + j; 

       if(j < (i+1)){
        lenperrow++;
        if(lenperrow == 1)
	 idxdisplace = idx2d;
       }
      
      //printf("i: %d j: %d  idx2d: %d array[idx2d]: %f\n", i, j, idx2d, aux);
     }
     arrayLen[i] = lenperrow;
     arrayDisplacement[i] = idxdisplace;
    }
}

