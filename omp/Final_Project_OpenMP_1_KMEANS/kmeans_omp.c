/*
 *
 * kMeans.c
 * 
 * Created on 28/9/22
 * Author Chris Kaldis
 * Version 0.0.4
 *  
 */

/*
    profiling

    module load intel impi
    icc -qopenmp kmeans_omp.c -o kmeans_omp

    ###ICC
    icc -g -pg -openmp-stubs -o kmeans_omp kmeans_omp.c
    icc -g -pg -qopenmp -o kmeans_omp kmeans_omp.c

    ls -ls gmon.out
    gprof -l kmeans_omp >kmeans_omp_gprof.out

    more kmeans_omp_gprof.out
    ###
    ###GCC
    gcc -o kmeans_omp –p -g kmeans_omp.c
    ./kmeans_omp
    gprof kmeans_omp gmon.out > kmeans_omp_gprof.out

    ###cc
    cc -o kmeans_omp kmeans_omp.c –fopenmp

    export 
    export OMP_NUM_THREADS=8
    ./kmeans_omp
*/

/*Incremental parallelization
    Largeapplications->focuson the most expensive parts

gprof report tell us that the most time consuming functions are:
distEucl (time 97%)
recalculateCenters (time 2%)
distEucl(time .2%)

SolutionExercise25.pdf
To calculate the speedups I used an interactive node with 16 cores, I measured the execution times 3 times for each configuration, 
    and calculated the average. The execution time of the sequential code was 35.1 s.

After adding #pragma omp parallel for reduction(+:distance) schedule(static,10)
    distEucl went down to 0.01s and time % to 0.00


Do NOT parallelize what does NOT matter
Identify opportunities to use the nowait clause
Parallel region cost incurred only once. Potential for the “nowait” clause
Do not share data unless you have to
Avoid nested parallelism, Consider tasking instead
Consider task loop as an alternative to a non-static loop iteration scheduling algorithm (e.g. dynamic)
False Sharing occurs when multiple threads modify the same cache line at the same time

An OpenMP place defines a set where a thread may run. OpenMP supports abstract names for places: sockets, cores, and threads

pragma omp for schedule(runtime)
export OMP_SCHEDULE=“dynamic,25”

# Use 2 sockets to place those threads:
export OMP_PLACES=sockets{2}

#Spread threads as close apart as possible:
export OMP_PROC_BIND=close


    lscpu
    numactl –H

        to reason about access time between nodes
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <omp.h>

#define N 100000 // N is the number of patterns
#define Nc 100 // Nc is the number of classes or centers
#define Nv 1000 // Nv is the length of each pattern (vector)
#define Maxiters 15 // Maxiters is the maximum number of iterations
#define Threshold 0.000001

double *mallocArray( double ***array, int n, int m, int initialize ) ;
void freeArray( double ***array, double *arrayData ) ;

void kMeans( double patterns[][Nv], double centers[][Nv] ) ;
void initialCenters( double patterns[][Nv], double centers[][Nv] ) ;
double findClosestCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***distances ) ;
void recalculateCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z ) ;

double distEucl( double pattern[], double center[] ) ;
int argMin( double array[], int length ) ;

void createRandomVectors( double patterns[][Nv] ) ;

int main( int argc, char *argv[] ) {

    double t1, t2;
    t1=omp_get_wtime();
	
    static double patterns[N][Nv] ;
	static double centers[Nc][Nv] ;

	createRandomVectors( patterns ) ;
	kMeans( patterns, centers ) ;

    t2=omp_get_wtime();

    printf("\n\n\nElapsed (tota) time: %15.15f seconds\n", t2 - t1);

    return EXIT_SUCCESS;    
}

/*
 *
 *
 * Create some random patterns for classification.
 * 
 * 
 */
//adding parallelism to the first for loop
void createRandomVectors( double patterns[][Nv] ) {

    double t1, t2;

	srand( 1059364 ) ;

	size_t i, j ;

    t1=omp_get_wtime();
    //All threads map to a place partition close to the parent thread
    //#pragma omp parallel private(i,j)
    //#pragma omp for collapse(2)
    
    //Each thread has a slice of “patterns” in its local memory
    //#pragma omp for schedule(static,10)
    for ( i = 0; i < N; i++ ) {
        for ( j = 0; j < Nv; j++ ) {
            patterns[i][j] = (double) (random()%100) - 0.1059364*(i+j) ;
        }
    }

    t2=omp_get_wtime();

    printf("\nTime taken to create random vector values is %g seconds\n\n", t2-t1);

	return ;
}

/*
 *
 *
 * Simple implementations of Lloyd's Algorithm
 *
 *
 */
void kMeans( double patterns[][Nv], double centers[][Nv] ) {

    double error = INFINITY ;
    double errorBefore ;
    int step = 0 ;
    
    // class or category of each pattern
    int *classes = (int *)malloc( N*sizeof(int) ) ;
    // distances between patterns and centers
    double **distances ;
    double *distanceData = mallocArray( &distances, N, Nc, 0 ) ;
    // tmp data for recalculating centers
    double **y, **z ;
    double *yData = mallocArray( &y, Nc, Nv, 1 ) ;
    double *zData = mallocArray( &z, Nc, Nv, 1 ) ;

    initialCenters( patterns, centers ) ; //step 1
    do {
        errorBefore = error ;
        error = findClosestCenters( patterns, centers, classes, &distances ) ; // step 2
        recalculateCenters( patterns, centers, classes, &y, &z ) ; // step 3
        printf( "Step:%d||Error:%lf,\n",step, (errorBefore-error)/error ) ;
        step ++ ;
    } while ( (step < Maxiters) && ((errorBefore-error)/error > Threshold) ) ; // step 4

    free( classes ) ;
    freeArray( &distances, distanceData ) ;
    freeArray( &y, yData ) ;
    freeArray( &z, zData ) ;

    return ;
}

/*
 *
 *
 * Allocates memory for a 2D array ([n][m]) of double type.
 * It uses malloc so if you want to initialize data
 * use initialize value != 0.
 *
 *
 */
double *mallocArray( double ***array, int n, int m, int initialize ) {

    * array = (double **)malloc( n * sizeof(double *) ) ;
    // avoid to fill heap with small memory allocations.
    double *arrayData = malloc( n*m * sizeof(double) ) ;

    if ( initialize != 0)
        memset( arrayData, 0, n*m ) ;
    
    size_t i ;

    double t1, t2;
    t1=omp_get_wtime();

    //#pragma omp parallel for private(i) shared(array) schedule(static,10) 
    for( i = 0; i < n; i++ )
        (* array)[i] = arrayData + i*m ;
    
    t2=omp_get_wtime();
    printf("\nTime taken to initialize 2D array ([n][m]) is %g seconds\n\n", t2-t1);
    
    return arrayData;
}


/*
 *
 *
 * Selects patterns (randomly) as initial centers,
 * different patterns used for each center.
 *
 *
 */
void initialCenters( double patterns[][Nv], double centers[][Nv] ) {

    int centerIndex ;
    size_t i, j ;


    double t1, t2;

    t1=omp_get_wtime();

    //Placing private for "j" here in order to reduce overhead of creating/destroying threads per (i * j) iterations
    //#pragma omp parallel for schedule(static,10) private(j)
    for ( i = 0; i < Nc; i++ ) {
        // split patterns in Nc blocks of N/Nc length
        // use rand and % to pick a random number of each block.
        centerIndex = rand()%( N/Nc*(i+1) - N/Nc*i + 1 ) + N/Nc*i ;
        for ( j = 0; j < Nv; j ++ ) {
                centers[i][j] = patterns[centerIndex][j] ;
        }
    }

    t2=omp_get_wtime();
    printf("\nTime taken to calculate initial centers is %g seconds\n\n", t2-t1);


    return ;
}

/*
 *
 *
 * Calculates the distance between patterns and centers,
 * then it finds the closest center for each pattern &
 * stores the index in the array named classes. Also
 * it calculates the error or the total distance
 * between the patterns and the closest mean-center.
 *
 *
 */
double findClosestCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***distances ) {

    double error = 0.0 ;
    size_t i, j ;

    //placing the parallel for here in order to reduce overhead of creating/destroying threads per (i * j) in the nested loop inside functions
    #pragma omp parallel for private(j) reduction(+:error) schedule(static)
    for ( i = 0; i < N; i++ ) {
            for ( j = 0; j < Nc; j++ )
                (* distances)[i][j] = distEucl( patterns[i], centers[j] ) ;
            classes[i] = argMin( (* distances)[i], Nc ) ;
            error += (* distances)[i][classes[i]] ;
        }
 

    return error;
}

/*
 *
 *
 * Finds the new means of each class using the patterns that
 * classified into this class.
 *
 *
 */
void recalculateCenters( double patterns[][Nv], double centers[][Nv], int classes[], double ***y, double ***z ) {

    //double error = 0.0 ;

    size_t i, j;

    double t1, t2;
    t1=omp_get_wtime();


    #pragma omp parallel
    {
        #pragma omp single
        //#pragma omp parallel for private(j) reduction(+:error)
        for ( i = 0; i < N; i++ ) {
            for ( j = 0; j < Nv; j++ ) {
                #pragma omp atomic
                (* y)[classes[i]][j] += patterns[i][j] ;
                #pragma omp atomic
                (* z)[classes[i]][j] ++ ;
            }
        }
    }
    //barrier to make sure that all threads have finished the calculation of tmp arrays
    
    //#pragma omp barrier
    // update step of centers
    for ( i = 0; i < Nc; i++ ) {
        for ( j = 0; j < Nv; j++ ) {
            centers[i][j] = (* y)[i][j]/(* z)[i][j] ;
            (* y)[i][j] = 0.0 ;
            (* z)[i][j] = 0.0 ;
        }
    }

    t2=omp_get_wtime();
    printf("\nTime taken for recalculateCenters is %g seconds\n\n", t2-t1);
    
    return ;
}

/*
 *
 *
 * Calclulates the Eucledean distance between a pattern
 * and a center.
 *
 *
 */
double distEucl( double pattern[], double center[] ) {

    double distance = 0.0 ;

    //#pragma omp parallel for reduction(+:distance) schedule(static,10)
    for ( int i = 0; i < Nv; i++ )
        distance += ( pattern[i]-center[i] )*( pattern[i]-center[i] ) ;
    
    return sqrt(distance) ;
}

/*
 *
 *
 * Finds the index of the minimum value of
 * an array with the current length.
 *
 *
 */
int argMin( double array[], int length ) {

    int index = 0 ;
    double min = array[0] ;

    for ( int i = 1; i < length; i++ ) {
        if ( min > array[i] ) {
            index = i ;
            min = array[i] ;
        }
    }

    return index ;
}

/*
 *
 *
 * Free memory of a 2D array of double type.
 * Memory allocated with the function mallocArray().
 *
 *
 */
void freeArray( double ***array, double *arrayData ) {
    
    free( arrayData ) ;
    free( * array ) ;

    return ;
}

