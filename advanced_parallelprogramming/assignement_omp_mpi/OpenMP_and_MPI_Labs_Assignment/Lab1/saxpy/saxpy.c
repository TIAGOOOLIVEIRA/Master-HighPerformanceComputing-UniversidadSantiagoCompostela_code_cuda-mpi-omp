// Compile with gcc -O2 -fopenmp -fopt-info-vec
// Compile with gcc -O2 -fopenmp -fopt-info-vec -march=native

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

void saxpy_no_simd( float *X, float *Y, int i, float a){
  Y[i] = a * X[i] + Y[i];
}

#pragma omp declare simd uniform(a)
void saxpy ( float *X, float *Y, int i, float a) {
  Y[i] = a * X[i] + Y[i];
}


float saxpyi_no_simd( float x, float y, float a){
  return a * x + y;
}

#pragma omp declare simd uniform(a)
float saxpyi( float x, float y, float a){
  return a * x + y;
}


#define N     128*131072
#define NREPS     100

float a[N] __attribute__ ((aligned(256)));
float b[N] __attribute__ ((aligned(256)));

double startTime, execTime;

double clock_it()
{ 
  struct timeval start;
  double duration;

  gettimeofday(&start, NULL);
  duration = (double)(start.tv_sec + start.tv_usec/1000000.0);
  return duration;
}

void init()
{ int i;
  
  for (i = 0; i < N; ++i) {
    a[i] = (float)(i + 1);
    b[i] = (float)i;
  }
}

float suma(int n, float *a)
{ int i;
  float r = 0;

  for (i = 0; i < n; ++i)
    r += a[i];
  return r;
}

int main()
{
  int i, j, k, n=N;

  init();
  startTime = clock_it();
  #pragma omp parallel for private(k, j)
  for (k = 0; k < NREPS;k++) {
    for(j= 0; j< N; j++){
      saxpy_no_simd(a, b, j, 4.0f);
    }
 }
  execTime = clock_it() - startTime;
  printf("saxpy_no_simd  %2.3f s. Result=%f\n", execTime, suma(n, a));


  init();
  startTime = clock_it();
  #pragma omp parallel for private(k, j)
  for (k = 0; k < NREPS;k++) {
    #pragma omp simd
    for(j= 0; j< N; j++){
      saxpy(a, b, j, 4.0f);
    }
 }
  execTime = clock_it() - startTime;
  printf("saxpy  %2.3f s. Result=%f\n", execTime, suma(n, a));

  
  init();
  startTime = clock_it();
  #pragma omp parallel for private(k, j)
  for (k = 0; k < NREPS;k++) {
    for(j= 0; j< N; j++){
      b[j] = saxpyi_no_simd(a[j], b[j], 4.0f);
    }
 }
  execTime = clock_it() - startTime;
  printf("saxpyi_no_simd  %2.3f s. Result=%f\n", execTime, suma(n, a));

  init();
  startTime = clock_it();
  #pragma omp parallel for private(k, j)
  for (k = 0; k < NREPS;k++) {
    #pragma omp simd
    for(j= 0; j< N; j++){
      b[j] = saxpyi(a[j], b[j], 4.0f);
    }
 }
  execTime = clock_it() - startTime;
  printf("saxpyi  %2.3f s. Result=%f\n", execTime, suma(n, a));




}