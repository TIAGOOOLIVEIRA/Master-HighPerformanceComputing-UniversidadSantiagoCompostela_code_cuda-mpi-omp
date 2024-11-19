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


const int N = 16;    // Número predeterm. de elementos en los vectores


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
void CR_CPU(float A[], float B[], float C[], float D[], float X[], const unsigned int n)
{
  
  unsigned ln=floor(log2(float(n)));
  int 		stride, step, k, l;
  float 	s1, s2;
  
  unsigned int numBytes = n * sizeof(float);
    
  stride = 2; step =1; k = n-1;
  // Forward elimination 
  
    for(int i = 0; i < ln-1; i++) {
    
    	for ( int j = step; j < n - 1; j=j+stride ){
		
		s1 = A[j]/B[j-step];
		s2 = C[j]/B[j+step]; 
		
		A[j] = - A[j-step]*s1;
		B[j] = B[j] - C[j-step] * s1 - A[j+step] * s2;
		C[j] = -C[j+step]*s2;
		D[j] = D[j] - D[j-step]*s1-D[j+step]*s2;
		
	}
	// last equation
	
	s1 = A[k]/B[k-step];
	s2 = 0; 
		
	A[k] = - A[k-step]*s1;
	B[k] = B[k] - C[k-step] * s1;
	D[k] = D[k] - D[k-step] * s1;
			
	 step = step + stride;	
      	 stride = stride *2;
	
    }
  
   // backward substitution
   
     k = n/2-1;
     l = n-1;
     s1 = (B[k] * B[l]) - (C[k]*A[l]);
     X[k] = (B[l]*D[k] - C[k]*D[l])/s1;
     
 
      X[l] = (D[l]*B[k] - D[k]*A[l])/s1;
     
      step = n/4; stride=n/2;
      k = step -1;
      
      for(int i = 0; i < ln-1; i++) {
      
               // First node
        	        X[k] = (D[k] - C[k]*X[k+step])/B[k];
	        
            	for ( int j = k+stride; j < n ; j=j+stride ){
		     X[j] = (D[j] - A[j]*X[j-step]-C[j]*X[j+step])/B[j];
	         }
	         
      	 step = step/2;
	 stride = stride/2;
	 k = step -1;
	 
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
  float *Xv = (float *) malloc(numBytes);
 
  Initialization(Av, Bv, Cv, Dv,  n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Reservar e inicializar vectores (%u)\n\n", n);


  // Sumamos vectores en CPU
  timestamp(&start);
       CR_CPU(Av, Bv, Cv, Dv, Xv,  n);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> CR en la  CPU  \n\n");

   free(Av);
   free(Bv);
   free(Cv);
   free(Dv);
   free(Xv);
  
  return(0);
}



