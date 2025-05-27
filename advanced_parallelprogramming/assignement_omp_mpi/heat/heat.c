/* Heat */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

double clock_it()
{ 
  struct timeval start;
  double duration;

  gettimeofday(&start, NULL);
  duration = (double)(start.tv_sec + start.tv_usec/1000000.0);
  return duration;
}

int main(argc,argv)
int argc;
char *argv[];
{
    int N=100000000, i, it;
    float *x, *y;
    double startTime, execTime, sum;
   
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
    #pragma omp parallel for simd
    for(i=0; i<N; i++){
    //	x[i] = (N/2.0 - i);
    //	y[i] = 0.0001*i;
	x[i] = 1.0;
	y[i] = 1.0;
    }

    /* Operation */

    startTime = clock_it();  
    sum = 0.;
    
    for (int it=0; it<1000; it++) { 
        #pragma omp parallel for simd schedule(static)
        for (int i=1; i<N-1; i++)
    	    y[i] = ( x[i-1]+x[i]+x[i+1] )/3.;
  	#pragma omp parallel for simd schedule(static)
	for (int i=1; i<N-1; i++)
    	    x[i] = y[i];
}
    execTime = clock_it() - startTime;
    printf("Tiempo  %2.3f s.\n", execTime);
    
    printf("Result = %g\n", x[0]);

    return 0;
}