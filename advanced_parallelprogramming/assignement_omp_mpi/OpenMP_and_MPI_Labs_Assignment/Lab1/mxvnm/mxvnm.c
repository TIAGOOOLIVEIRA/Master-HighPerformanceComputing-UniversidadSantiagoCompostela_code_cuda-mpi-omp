 /* Matrix-vector product. Matrix of N*M */

 #include <stdio.h>
 #include <stdlib.h>
 
 int main(int argc,char *argv[])
 {
     int i, j, N, M;
     float **A, *Avector, *x, *y, temp;
 
     if (argc < 3) {
          fprintf(stderr,"Usar: %s filas columnas\n", argv[0]);
          exit(EXIT_FAILURE);
     }
 
     N = atoi(argv[1]);  //Rows of the matrix and elements of vector y
     M = atoi(argv[2]);  //Columns of the matrix and elements of vector x
 
       /* Allocate memory for global matrix */
    if((Avector = (float *) malloc(N*M*sizeof(float))) == NULL)
         printf("Error en malloc Avector[%d]\n",N*M);
 
       /* Pointer array for addressing A as a matrix */
    if((A = (float **) malloc(N*sizeof(float *))) == NULL)
         printf("Error en malloc del array de %d punteros\n",N);
       /* Asign to access A as a matrix: each pointer pointers to the start of the row */
       /* This way, all rows are consecutively in memory */
    for(i=0;i<N;i++)
       *(A+i) = Avector+i*M;
 
       /* Matrix inicialization */
    for(i=0; i<N; i++)
     for(j=0; j<M; j++)
         A[i][j] = (0.15*i - 0.1*j)/N;
    
 
       /* Allocate memory for x and y vectors */
     if((x = (float *) malloc(M*sizeof(float))) == NULL)
         printf("Error en malloc x[%d]\n",M);
     if((y = (float *) malloc(N*sizeof(float))) == NULL)
         printf("Error en malloc y[%d]\n",N);
 
       /* x vector inicialization */
     for(i=0; i<M; i++)
     x[i] = (M/2.0 - i);
 
 
     /* Matrix-vector product, y = Ax */    
     for(i=0; i<N; i++){
     temp = 0.0;
     for(j=0; j<M; j++)
         temp += A[i][j]*x[j];
     y[i] = temp;
     }
         
     printf("Done,  y[0] = %g  y[%d] = %g \n", y[0], N-1, y[N-1]);
 
     return 0;
 }
 
 
 
 