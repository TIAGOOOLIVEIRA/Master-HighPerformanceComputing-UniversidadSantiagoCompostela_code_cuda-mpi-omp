/* 
   Programación de GPUs (General Purpose Computation on Graphics Processing 
   Unit)

   PCR en GPU
   Parámetros opcionales (en este orden): sumavectores #rep #n #blk
   #rep: número de repetiones
   #n: número de elementos en cada vector
   #blk: hilos por bloque CUDA
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


const int N = 1024;    // Número predeterm. de elementos en los vectores

const int CUDA_BLK = 16;  // Tamaño predeterm. de bloque de hilos ƒCUDA


/* 
   Para medir el tiempo transcurrido (elapsed time):

   resnfo: tipo de dato definido para abstraer la métrica de recursos a usar
   timenfo: tipo de dato definido para abstraer la métrica de tiempo a usar

   timestamp: abstrae función usada para tomar las muestras del tiempo transcurrido

   printtime: abstrae función usada para imprimir el tiempo transcurrido

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): función para obtener 
   el tiempo transcurrido entre dos medidas
*/

#include <sys/time.h>
#include <sys/resource.h>

#ifdef _noWALL_
typedef struct rusage resnfo;
typedef struct _timenfo {
  double time;
  double systime;
} timenfo;
#define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
#define printtime(t) printf("%15f s (%f user + %f sys) ",		\
			    t.time + t.systime, t.time, t.systime);
#else
typedef struct timeval resnfo;
typedef double timenfo;
#define timestamp(sample)     gettimeofday((sample), 0)
#define printtime(t) printf("%15f s ", t);
#endif

void myElapsedtime(const resnfo start, const resnfo end, timenfo *const t)
{
#ifdef _noWALL_
  t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6)) 
    - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
  t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6)) 
    - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
#else
  *t = (end.tv_sec + (end.tv_usec * 1E-6)) 
    - (start.tv_sec + (start.tv_usec * 1E-6));
#endif /*_noWALL_*/
}


/*
  Función para inicializar los vectores que vamos a utilizar
*/
void Initialization(float A[], float B[], float C[], float D[], const unsigned int n)
{
  unsigned int i;

     A[0] = 0.0; B[0] = 2.0; C[0] = -1.0; D[0] = 1.0;
     
  for(i = 1; i < n-1; i++) {
    A[i] = -1.0;
    B[i] = 2.0;
    C[i] = -1.0;
    D[i] = 0.0;
  }
  
  A[n-1] = -1.0; B[n-1] = 2.0; C[n-1] = 0.0; D[n-1] = 1.0;
}



/*
  Función PCR en la CPU
*/
void PCR_CPU(float X[], float Y[], float Z[], float W[], const unsigned int n)
{
  unsigned int i, k;
  unsigned ln=floor(log2(float(n)));
  float 	alpha, gamma;
  
  unsigned int numBytes = n * sizeof(float);

  float *Xr = (float *) malloc(numBytes);
  float *Yr = (float *) malloc(numBytes);
  float  *Zr = (float *) malloc(numBytes);
  float *Wr = (float *) malloc(numBytes);
  
  k = 1;
    for(i = 0; i < ln; i++) {
    	for ( int j = 0; j < n; j++){
		
		if( j >= k ){
		
			
			if(  j <= (n-k-1) ){
			
          			alpha = -X[j] / Y[j-k];
            			gamma = - Z[j] / Y[j+k];

           			Yr[j] = Y[j] + (alpha * Z[j-k] + gamma * X[j+k]);
            			Xr[j] = alpha * X[j-k];
            			Zr[j] = gamma * Z[j+k];
            			Wr[j] = W[j] + (alpha * W[j-k] + gamma * W[j+k]);					
			}
			else{
      				alpha = -X[j] / Y[j-k];
            			
           			Yr[j] = Y[j] + (alpha * Z[j-k] );
            			Xr[j] = alpha * X[j-k];
            			Zr[j] = 0;
            			Wr[j] = W[j] + (alpha * W[j-k] );					
			}
		}
		else{
		
            			gamma = -Z[j] / Y[j+k];

           			Yr[j] = Y[j] + gamma * X[j+k];
            			Xr[j] = 0;
            			Zr[j] = gamma * Z[j+k];
            			Wr[j] = W[j] + gamma * W[j+k];
			
			}
		
      	
	}
	k = k << 1;
	for ( int j = 0; j < n; j++){
		X[j] = Xr[j];
		Y[j] = Yr[j];
		Z[j] = Zr[j];
		W[j] = Wr[j];
	}
    }
  
  	for ( int j = 0; j < n/2; j++){
		float temp;
	        
		temp = Y[j+n/2]*Y[j] - Z[j]*X[j+n/2] ;
		X[j] = (Y[j+n/2] * W[j] - Z[j] * W[j+n/2]) / temp;
		X[j+n/2] = (W[j+n/2] * Y[j] - W[j] * X[j+n/2]) / temp;
	}

	
	for ( int j = 0; j < n; j++){
		printf(" \t %f  \n", X[j]);
	}
  
}



/*
  Función principal
*/
int main(int argc, char *argv[])
{
  // Para medir tiempos
  resnfo start, end, startgpu, endgpu;
  timenfo time, timegpu;

  // Aceptamos algunos parámetros

  // Número de elementos en los vectores (predeterminado: N)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;

  // Número de bytes a reservar para nuestros vectores
  unsigned int numBytes = n * sizeof(float);

  // Reservamos e inicializamos vectores
  timestamp(&start);
  float *Av = (float *) malloc(numBytes);
  float *Bv = (float *) malloc(numBytes);
  float *Cv = (float *) malloc(numBytes);
  float *Dv = (float *) malloc(numBytes);
 
  Initialization(Av, Bv, Cv, Dv,  n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Reservar e inicializar vectores (%u)\n\n", n);


  // Sumamos vectores en CPU
  timestamp(&start);
  PCR_CPU(Av, Bv, Cv, Dv,  n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> PCR en la  CPU  \n\n");

   free(Av);
   free(Bv);
   free(Cv);
   free(Dv);
  
  return(0);
}




