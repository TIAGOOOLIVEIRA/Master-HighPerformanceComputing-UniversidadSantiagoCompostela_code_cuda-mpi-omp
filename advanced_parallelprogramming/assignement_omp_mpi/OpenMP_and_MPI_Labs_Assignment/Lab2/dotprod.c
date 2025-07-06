/* Dot product of two vectors */

#include <stdio.h>
#include <stdlib.h>

int main(argc,argv)
int argc;
char *argv[];
{
    int N=100000000, i;
    float *x, *y;
    float dot;
    
    if (argc < 2) {
     	fprintf(stderr,"Use: %s num_elem_vector\n", argv[0]);
     	exit(EXIT_FAILURE);
    }

    N = atoi(argv[1]);

      /* Allocate memory for vectors */
    if((x = (float *) malloc(N*sizeof(float))) == NULL)
        printf("Error in malloc x[%d]\n",N);
    if((y = (float *) malloc(N*sizeof(float))) == NULL)
        printf("Error in malloc y[%d]\n",N);

      /* Inicialization of x and y vectors*/
    for(i=0; i<N; i++){
	x[i] = (N/2.0 - i);
	y[i] = 0.0001*i;
    }

    /* Dot product operation */

    
    dot = 0.;
    for(i=0; i<N; i++)
	dot += x[i] * y[i];
  
    printf("Dot product = %g\n", dot);

    return 0;
}



