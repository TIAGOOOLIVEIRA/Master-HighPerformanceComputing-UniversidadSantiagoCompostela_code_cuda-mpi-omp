#include <stdio.h>
#include <omp.h>
/* Program for multiplication of D=A*B */

#define NRA 2048		/* number of rows in matrix A */
#define NCA 2048        	/* number of columns in matrix A */
#define NCB 2048		/* number of columns in matrix B */
static float a[NRA][NCA] __attribute__ ((aligned(256)));    /* matrix A to be multiplied */
static float b[NCA][NCB] __attribute__ ((aligned(256)));     	/* matrix B to be multiplied */
static float d[NRA][NCB] __attribute__ ((aligned(256)));      	/* result matrix A*B */

void main()
{
int    i, j, k;			/* misc */

double start, end;
float temp;


   /* Initialize A, B, D*/
   for (i=0; i< NRA; i++)
      for (j=0; j< NCA; j++)
         a[i][j]= 1.;
   for (i=0; i< NCA; i++)
      for (j=0; j< NCB; j++)
         b[i][j]= 1.;

   for(i=0;i< NRA;i++)
      for(j=0;j< NCB;j++)
         d[i][j] = 0.0;



   start = omp_get_wtime();
   /* Perform matrix multiply A.BT */
   #pragma omp parallel for private(i, j, k, temp) collapse(2)
   for(i=0;i< NRA;i++)
      for(j=0;j< NCB;j++)
      {
	 temp = 0.0;
         #pragma omp simd reduction(+:temp)
	 for(k=0;k< NCA;k++)
           temp += a[i][k] * b[j][k];
	 d[i][j] = temp;
      }
   end = omp_get_wtime();

   printf("Done\n");

   printf("d[0][0]= %f\n ", d[0][0]);
   printf("d[NRA-1][NCB-1]= %f\n ", d[NRA-1][NCB-1]);
   printf("Execution time = %g seconds\n ", end-start);
   printf ("\n");
}


// Compile with gcc -O2 -fopenmp -fopt-info-vec
// Compile with gcc -O2 -fopenmp -fopt-info-vec -march=native
// gcc -O2 -fno-tree-vectorize -fopenmp -fopt-info-vec  disables vectorization 



/* 
   To compile: gcc -O2 -fopenmp -fopt-info-vec multf.c
   To run: ./a.out
   To see the vectorization report: ./a.out > report.txt
   To see the vectorization report with optimization level 2: gcc -O2 -fopenmp -fopt-info-vec multf.c > report.txt
   To see the vectorization report with optimization level 3: gcc -O3 -fopenmp -fopt-info-vec multf.c > report.txt
*/
/* 
   To compile with march=native: gcc -O2 -fopenmp -fopt-info-vec -march=native multf.c
   To run: ./a.out
   To see the vectorization report: ./a.out > report.txt
   To see the vectorization report with optimization level 2: gcc -O2 -fopenmp -fopt-info-vec -march=native multf.c > report.txt
   To see the vectorization report with optimization level 3: gcc -O3 -fopenmp -fopt-info-vec -march=native multf.c > report.txt
*/