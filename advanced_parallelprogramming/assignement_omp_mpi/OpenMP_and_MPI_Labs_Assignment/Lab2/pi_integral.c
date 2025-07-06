/* Obtain the PI value using the numerical integration of 4/(1+x*x) between 0 and 1.
The numerical integration is calculated with n rectangular intervals (area=(1/n)*4/(1+x*x))) and adding the area of all these rectangle intervals  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(argc,argv)
int argc;
char *argv[];
{
    long int n=1000000000, i;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;

    if (argc < 2) {
     	fprintf(stderr,"Usar: %s num_iteraciones\n", argv[0]);
     	exit(EXIT_FAILURE);
    }

    n = atol(argv[1]); //num. of intervals
  
    h   = 1.0 / (double) n;  //wide of the rectangle
    sum = 0.0;
    for (i = 0; i < n; i ++) {
	x = h * ((double)i + 0.5);   //height of the rectangle
        sum += 4.0 / (1.0 + x*x);
    }
    pi = h * sum;

    printf("The obtained Pi value is: %.16f, the error is: %.16f\n", pi, fabs(pi - PI25DT));

    return 0;
}
