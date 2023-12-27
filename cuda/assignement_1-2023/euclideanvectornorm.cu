/* 
   Programación de GPUs (General Purpose Computation on Graphics Processing 
   Unit)
©
   Margarita Amor López
   Emilio J. Padrón González

   sumavectores.c

   Suma dos vectores en CPU y GPU
   Parámetros opcionales (en este orden): sumavectores #rep #n #blk
   #rep: número de repetiones
   #n: número de elementos en cada vector
   #blk: hilos por bloque CUDA

###########################################################################################
Student: Tiago de Souza Oliveira
Assignment 1 - 2023
           Multiple euclidean vector norme on a GPU

Taking  the sumavectores.cu code in order to reuse cross cutting functions in my solution




To compile, run and profile:
    compute --gpu
    nvcc -g -G -c euclideanvectornorm.cu -o euclideanvectornorm.o
    sudo nvpp ./euclideanvectornorm
    sudo nvprof --unified-memory-profiling off ./euclideanvectornorm
    nvcc -o euclideanvectornorm euclideanvectornorm.cu 
    
    ./euclideanvectornorm
    
    sbatch job_euclideanvectornorm.sh
    watch -n 1 squeue -u curso370


v1:
The core functions I created to tackle that are listed below:
    1. fill_matrix: Responsibe to create a 2D matrix filled with int ascending values starting from 1
        adding +1 to upcoming position until n size of the dimension 
    2. power_matrix_kernel_cuda: Responsible to calculate the power value for each element in the matrix on GPU
    3. reduce_sum_vector_kernel_cuda: Responsible to summarize each row in the matrizx and return a vector with
        the reduced value per row on GPU
    4. power_matrix_GPU: responsible to manage memory de/allocation for CPU/GPU, block/grid definition and 
        profiling GPU execution as well for power computing
    5. sum_vector_GPU: responsible to manage memory de/allocation for CPU/GPU, block/grid definition and
        profiling GPU execution as well for sum vector elements computing
v2:
###########################################################################################
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

const int N = 524288;    // Standard value for n dimension in the matrix

const int M = 524288;    // Standard value for m dimension in the matrix


const int CUDA_BLK = 1024;
//On current GPUs, a thread block may contain up to 1024 threads
//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction

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
#define printtime(t) printf("%15f s (%f user + %f sys) ",               \
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
   Function to fill a matrix with ascending values starting from 1
   adding +1 to upcoming position until n size of the dimension
*/
void fill_vector_2dim(const Matrix mA){
  printf("\n\n To Fill Matrix width (%u) height(%u) \n", mA.width, mA.height);
  for(unsigned int i = 0; i < mA.height; i++) {
   for(unsigned int j = 0; j < mA.width; j++) {
      unsigned int idx2d = i * mA.width + j;
      float v = (float) (j + (i+1));
      //printf("\n width (%u) height(%u) i(%u) j(%u) idx2d(%u) value(%f)", mA.width, mA.height, i, j, idx2d, v);
      mA.elements[idx2d] = v;
   }
  }
}


/*
   Declaration of a function responsible to receive a matrix mA, summarize elements per row to reduce value into a vector mR 
   Each thread from CUDA GPU should take care of each row to perform the sum operation
*/
__global__ void reduce_sum_vector_kernel_cuda_struct(const Matrix mA,const Matrix mR);

__global__ void power_reducesum_vector_kernel_cuda_struct(const Matrix mA,const Matrix mR);

/*
   Declaration of a function responsible to receive a matrix mA, perform power calculation on each element andd update related element position with the new value
   Each thread from CUDA GPU should take care of each element to perform the power calculation
*/
__global__ void power_matrix_kernel_cuda_struct(const Matrix mA, const Matrix mR);

void devicenfo(void);

// Declaración de función para comprobar y ajustar los parámetros de
// ejecución del kernel a las restricciones de la GPU
void checkparams(unsigned int *n, unsigned int *m, unsigned int *cb);


void power_sum_matrix_GPU_2kernell_GPUmatrix(const Matrix mA, const Matrix mPowerA, const unsigned int blk_size, resnfo *const start, resnfo *const end){
  printf("\nInitializing power_sum_matrix_GPU_2kernell_GPUmatrix");
  unsigned int numBytes = mA.width * mA.height * sizeof(float);

  //to allocate memory space on GPU and copy matrix from CPU to GPU
  Matrix d_A;
  d_A.width = mA.width; 
  d_A.height = mA.height;
  cudaMalloc(&d_A.elements, numBytes);
  cudaMemcpy(d_A.elements, mA.elements, numBytes, cudaMemcpyHostToDevice);

  Matrix d_RPowerMatrix;
  //should match the length of d_A height and width 1 (a vector)
  d_RPowerMatrix.width = mA.width; 
  d_RPowerMatrix.height = mA.height;
  cudaMalloc(&d_RPowerMatrix.elements, numBytes);  
  cudaMemset(d_RPowerMatrix.elements, 0, numBytes);

  Matrix d_R;
  //should match the length of d_A height and width 1 (a vector)
  d_R.width = 1; 
  d_R.height = d_A.height;
  unsigned int numBytesR = d_R.width * d_R.height * sizeof(float);
  cudaMalloc(&d_R.elements, numBytesR);  
  cudaMemset(d_R.elements, 0, numBytesR);


  printf("\nAllocated into GPU memory inout Matrix\n");

  //definiting block of threads for 2D matrix
  //Number of threads in each dimension of the blocks (blockDim.x, blockDim.y, blockDim.z)
  //dim3 dimBlock(blk_size, blk_size);
  dim3 dimBlock(blk_size);

  printf("\nCalling kernel power_matrix_kernel_cuda_struct for mA.width(%u) and mA.height(%u) dimBlock.x %u dimBlock.y %u \n", mA.width, mA.height, dimBlock.x, dimBlock.y);

  //Number of blocks in each dimension of the grid. (gridDim.x, gridDim.y)
  //dim3 dimGrid((mA.width + dimBlock.x - 1) / dimBlock.x, (mA.height + dimBlock.y - 1) / dimBlock.y);
  dim3 dimGrid(((mA.width * mA.height) + dimBlock.x - 1)/dimBlock.x);


  timestamp(start);


  //to call kernel responsible to power each element and reduce sum rows into element in a vector
  //power_reducesum_vector_kernel_cuda_struct<<<dimGrid, dimBlock>>>(d_A, d_R);
  power_matrix_kernel_cuda_struct<<<dimGrid, dimBlock>>>(d_A, d_RPowerMatrix);

  cudaDeviceSynchronize();

  //call kernel responsible to reduce sum each row from powered matrix with power matrix on GPU memory
  reduce_sum_vector_kernel_cuda_struct<<<dimGrid, dimBlock>>>(d_RPowerMatrix, d_R);


  cudaDeviceSynchronize();

  timestamp(end);

  //Copy from GPU the matrix with the elements power calculated to use in the subsequent step: Sum elements on each vector
  cudaMemcpy(mPowerA.elements, d_R.elements, numBytesR, cudaMemcpyDeviceToHost); // GPU -> CPU


  printf("\n\nOriginal Matrix:");
  for(unsigned int i = 0; i < mA.height; i++){
    printf("\n");
    for(unsigned int j = 0; j < mA.width; j++){
      unsigned int idx2d = i * mA.width + j;
      printf("%f,", mA.elements[idx2d]);
    }
  } 


  cudaFree(d_A.elements);
  cudaFree(d_R.elements);
  cudaFree(d_RPowerMatrix.elements);


      //to validate reduce sum vector
      printf("\n\nReduce sum vector: mR.height(%u) mR.width(%u)", mPowerA.height, mPowerA.width);
      for(unsigned int i = 0; i < mPowerA.height; i++){
        for(unsigned int j = 0; j < mPowerA.width; j++){
          unsigned int idx2d = i * mPowerA.width + j;
          printf("\n");
          printf("%f,", mPowerA.elements[idx2d]);
        }
      } 

}


void power_sum_matrix_GPU(const Matrix mA, const Matrix mPowerA, const unsigned int blk_size, resnfo *const start, resnfo *const end){
  printf("\nInitializing power_sum_matrix_GPU");
  unsigned int numBytes = mA.width * mA.height * sizeof(float);

  //to allocate memory space on GPU and copy matrix from CPU to GPU
  Matrix d_A;
  d_A.width = mA.width; 
  d_A.height = mA.height;
  cudaMalloc(&d_A.elements, numBytes);
  cudaMemcpy(d_A.elements, mA.elements, numBytes, cudaMemcpyHostToDevice);

  Matrix d_R;
  //should match the length of d_A height and width 1 (a vector)
  d_R.width = 1; 
  d_R.height = d_A.height;
  unsigned int numBytesR = d_R.width * d_R.height * sizeof(float);
  cudaMalloc(&d_R.elements, numBytesR);  
  cudaMemset(d_R.elements, 0, numBytesR);


  printf("\nAllocated into GPU memory inout Matrix\n");

  //definiting block of threads for 2D matrix
  //Number of threads in each dimension of the blocks (blockDim.x, blockDim.y, blockDim.z)
  //dim3 dimBlock(blk_size, blk_size);
  dim3 dimBlock(blk_size);


  //Number of blocks in each dimension of the grid. (gridDim.x, gridDim.y)
  //dim3 dimGrid((mA.width + dimBlock.x - 1) / dimBlock.x, (mA.height + dimBlock.y - 1) / dimBlock.y);
  dim3 dimGrid(((mA.width * mA.height) + dimBlock.x - 1)/dimBlock.x);

  timestamp(start);

  printf("\nCalling kernel power_reducesum_vector_kernel_cuda_struct for mA.width(%u) and mA.height(%u) dimBlock.x %u dimBlock.y %u \n", mA.width, mA.height, dimBlock.x, dimBlock.y);
  //to call kernel responsible to power each element and reduce sum rows into element in a vector
  power_reducesum_vector_kernel_cuda_struct<<<dimGrid, dimBlock>>>(d_A, d_R);
  printf("\nCalled kernel power_reducesum_vector_kernel_cuda_struct for mA.width(%u) and mA.height(%u) dimBlock.x %u dimBlock.y %u \n", mA.width, mA.height, dimBlock.x, dimBlock.y);


  cudaDeviceSynchronize();

  timestamp(end);

  //Copy from GPU the matrix with the elements power calculated to use in the subsequent step: Sum elements on each vector
  cudaMemcpy(mPowerA.elements, d_R.elements, numBytesR, cudaMemcpyDeviceToHost); // GPU -> CPU


  printf("\n\nOriginal Matrix:");
  for(unsigned int i = 0; i < mA.height; i++){
    printf("\n");
    for(unsigned int j = 0; j < mA.width; j++){
      unsigned int idx2d = i * mA.width + j;
      printf("%f,", mA.elements[idx2d]);
    }
  } 


  cudaFree(d_A.elements);
  cudaFree(d_R.elements);


      //to validate reduce sum vector
      printf("\n\nReduce sum vector: mR.height(%u) mR.width(%u)", mPowerA.height, mPowerA.width);
      for(unsigned int i = 0; i < mPowerA.height; i++){
        for(unsigned int j = 0; j < mPowerA.width; j++){
          unsigned int idx2d = i * mPowerA.width + j;
          printf("\n");
          printf("%f,", mPowerA.elements[idx2d]);
        }
      } 

}

/*
   Function to manage memory allocation from host to device and the design of blocks of threads on GPU
   having each thread to  perform power operation on each element in the matrix mA and update back each value to matrix
*/
void power_matrix_GPU(const Matrix mA, const Matrix mPowerA, const unsigned int blk_size, resnfo *const start, resnfo *const end){
  //size of the matrix: 2D => m x n
  printf("\nInitializing power_matrix_GPU");
  
  unsigned int numBytes = mA.width * mA.height * sizeof(float);

  //to allocate memory space on GPU and copy matrix from CPU to GPU
  Matrix d_A;
  d_A.width = mA.width; 
  d_A.height = mA.height;
  cudaMalloc(&d_A.elements, numBytes);
  cudaMemcpy(d_A.elements, mA.elements, numBytes, cudaMemcpyHostToDevice);

  Matrix d_R;
  //should match the length of d_A height and width 1 (a vector)
  d_R.width = mA.width; 
  d_R.height = mA.height;
  cudaMalloc(&d_R.elements, numBytes);  
  cudaMemset(d_R.elements, 0, numBytes);

  printf("\nAllocated into GPU memory inout Matrix\n");

  //definiting block of threads for 2D matrix
  //Number of threads in each dimension of the blocks (blockDim.x, blockDim.y, blockDim.z)
  //dim3 dimBlock(blk_size, blk_size);
  dim3 dimBlock(blk_size);

  printf("\nCalling kernel power_matrix_kernel_cuda_struct for mA.width(%u) and mA.height(%u) dimBlock.x %u dimBlock.y %u \n", mA.width, mA.height, dimBlock.x, dimBlock.y);

  //Number of blocks in each dimension of the grid. (gridDim.x, gridDim.y)
  //dim3 dimGrid((mA.width + dimBlock.x - 1) / dimBlock.x, (mA.height + dimBlock.y - 1) / dimBlock.y);
  dim3 dimGrid(((mA.width * mA.height) + dimBlock.x - 1)/dimBlock.x);


  timestamp(start);
  power_matrix_kernel_cuda_struct<<<dimGrid, dimBlock>>>(d_A, d_R);

  //to ensure all GPU´s threads have finished its processing 
  cudaDeviceSynchronize();

  timestamp(end);

  //Copy from GPU the matrix with the elements power calculated to use in the subsequent step: Sum elements on each vector
  cudaMemcpy(mPowerA.elements, d_R.elements, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU

  cudaFree(d_A.elements);
  cudaFree(d_R.elements);

  //to validate matrix input elements
  /*
  printf("Original Matrix:");
  for(unsigned int i = 0; i < mA.height; i++){
    printf("\n");
    for(unsigned int j = 0; j < mA.width; j++){
      unsigned int idx2d = i * mA.width + j;
      printf("%f,", mA.elements[idx2d]);
    }
  } 

  //to validate power elements matrix
  printf("\nPower Matrix:");
  for(unsigned int i = 0; i < mPowerA.height; i++){
    printf("\n");
    for(unsigned int j = 0; j < mPowerA.width; j++){
      unsigned int idx2d = i * mPowerA.width + j;
      printf("%f,", mPowerA.elements[idx2d]);
    }
  }   
  */
   
   /*
   dim3 gridDim -> dimensions of the grid
   uint3 blockIdx -> index of the block within the grid
   dim3 blockDim -> dimensions of the block
   uint3 threadIdx -> index of the thread or task within the block
   int warpSize -> size of the warp (measured in number of tasks)
   */
}

/*
   Function to manage memory allocation from host to device and the design of blocks of threads on GPU
   to summarize elements in vectors of Matrix and return a vector of the related reduced value
*/
void sum_vector_GPU(const Matrix mA, const Matrix mR, const unsigned int blk_size, resnfo *const start, resnfo *const end){
  //size of the matrix: 2D => m x n

  Matrix d_A;
  d_A.width = mA.width; 
  d_A.height = mA.height;
  unsigned int numBytes = mA.width * mA.height * sizeof(float);
  cudaMalloc(&d_A.elements, numBytes);
  cudaMemcpy(d_A.elements, mA.elements, numBytes, cudaMemcpyHostToDevice);

  Matrix d_R;
  //should match the length of d_A height and width 1 (a vector)
  d_R.width = 1; 
  d_R.height = d_A.height;
  unsigned int numBytesR = d_R.width * d_R.height * sizeof(float);
  cudaMalloc(&d_R.elements, numBytesR);  
  cudaMemset(d_R.elements, 0, numBytesR);
  
  //definiting block of threads, where each thread takes the responsibility to reduce in sum operation all elements on each row from powered Matrix
  //Number of threads in each dimension of the blocks (blockDim.x, blockDim.y, blockDim.z)
  //One=dimensional block of threads (* blk size * threads)
  dim3 dimBlock(blk_size);
  
  //dim3 dimGrid((d_A.height + dimBlock.x - 1) / dimBlock.x);
  dim3 dimGrid(((mA.width * mA.height) + dimBlock.x - 1)/dimBlock.x);
  timestamp(start);
  
  reduce_sum_vector_kernel_cuda_struct<<<dimGrid, dimBlock>>>(d_A, d_R);

  //to ensure all threads have finished its processing 
  cudaDeviceSynchronize();
  
  timestamp(end);
  
  cudaMemcpy(mR.elements, d_R.elements, numBytesR, cudaMemcpyDeviceToHost); // GPU -> CPU
  
  cudaFree (d_A.elements);
  cudaFree (d_R.elements);

    //to validate reduce sum vector
    printf("\nReduce sum vector: mR.height(%u) mR.width(%u)", mR.height, mR.width);
    /*
    for(unsigned int i = 0; i < mR.height; i++){
      for(unsigned int j = 0; j < mR.width; j++){
        unsigned int idx2d = i * mR.width + j;
        printf("\n");
        printf("%f,", mR.elements[idx2d]);
      }
    } 
    */
} 


int main(int argc, char *argv[])
{
  resnfo start, end, startgpu, endgpu;
  timenfo time, timegpu;

  // size of dimension n => number of rows (Global variable predefined: N)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;
  // size of dimension m => number of columns (Global variable predefined: m)
  unsigned int m = (argc > 1)?atoi (argv[2]):M;

  //in case a dimension is not defined, the application snould finish
  if (n == 0 || m == 0) {
    devicenfo();
    return(0);
  }

   // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK)
  unsigned int cb = (argc > 2)?atoi (argv[3]):CUDA_BLK;

  checkparams(&n, &m, &cb);

  printf("\n\n Matrix width (%u) height(%u) blocksize(%u) \n\n", m, n, cb);


  // Number of bytes to allocate to the matrix
  unsigned int numBytes = n * m * sizeof(float);

  // To allocate the filled matrix in CPU memory
  timestamp(&start);
  
  //to allocate memory space on CPU
  //Matrix matrixA[n][m] represented as unidimensional array as input for the power calculation over each element
  Matrix d_A;
  d_A.width = m; 
  d_A.height = n;
  d_A.elements = (float *) malloc(numBytes);

  //to allocate memory space on CPU
  //Matrix matrixR[n][m] represented as unidimensional array as output for the power calculation over each element
  Matrix d_R;
  d_R.width = m; 
  d_R.height = n;
  d_R.elements = (float *) malloc(numBytes);
  
  //vector result to store the sum of rows from powered Matrix
  //should match the length of d_A height and width 1 (a vector)
  Matrix v_R;
  v_R.width = 1; 
  v_R.height = n;
  unsigned int numBytesR = v_R.width * v_R.height * sizeof(float);
  v_R.elements = (float *) malloc(numBytesR);

  //to allocate memory space for the matrix in CPU
  fill_vector_2dim(d_A);

  timestamp(&end);

  myElapsedtime(start, end, &time);

  printf("\n\n -> to allocate and fill matrix n(%u) m(%u) blocksize(%u) \n\n", n, m, cb);

  printtime(time);
  printf(" -> to allocate and fill matrix n(%u) m(%u) blocksize(%u) \n\n", n, m, cb);


  //#######################Step 1: Power each element in the Matrix###########################################################################################
  // Perform the power calculation over matrix elements on GPU
  printf("Computing on GPU for Power elements in the Matrix \n");

  timestamp(&start);
  //power_matrix_GPU(d_A, d_R, cb, &startgpu, &endgpu);
  power_sum_matrix_GPU(d_A, v_R, cb, &startgpu, &endgpu);
  //power_sum_matrix_GPU_2kernell_GPUmatrix(d_A, v_R, cb, &startgpu, &endgpu);
  timestamp(&end);

  myElapsedtime(start, end, &time);

  printf("\n\nFinished computing on GPU: Total time CPU<-->GPU \n");
  printtime(time);
  printf(" -> Power matrix elements on GPU \n");

  // Separating time between GPU processing and data transferring CPU <-> GPU
  myElapsedtime(startgpu, endgpu, &timegpu);

  printtime(timegpu);
  printf(" -> Processing time on GPU only for calculating power on each element in the Matrix\n\t\t%15f s alloc and comm\n", time - timegpu);

  //######################Step 2: Reduce Sum on rows, after power processing on matrix elements###############################################################
  //Processing reduce sum over rows in the Powered Matrix on GPU
  //each thread should be assigned to each row to perform the reduce sum and persist the 
  //result in th vector 
  /*
  timestamp(&start);
  sum_vector_GPU(d_R, v_R, cb, &startgpu, &endgpu);
  timestamp(&end);

  myElapsedtime(start, end, &time);
  printtime(time);
  printf(" -> Reduce sum vectors on GPU \n\n");

  // Separating time between GPU processing and data transferring CPU <-> GPU
  myElapsedtime(startgpu, endgpu, &timegpu);

  printtime(timegpu);
  printf("Processing time on GPU for reduce sum on each row from power element Matrix\n\t\t%15f s alloc and comm\n", time - timegpu);
  */
  free(d_A.elements);
  free(d_R.elements);
  free(v_R.elements);

  return(0);
}

/*
Kernel implementation for reduce sum on each row from powered Matrix
It receives a matrix mA and assignd each thread to each row to perform the reduce sum and store the result in th vector mR
*/
__global__ void power_reducesum_vector_kernel_cuda_struct(const Matrix mA,const Matrix mR){
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;
    unsigned int idx = row * mA.width + col;

  float r = 0;
 
  //to ensure no thread will point outside matrix boundary (height)
  if(idx < mA.height){
    //to reduce as a summarize operation on elements of a row from the power matrix
   for(unsigned int j = 0; j < mA.width; j++){
    unsigned int idx2d = col * mA.width + j;
    //https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html
    float vp = __powf(mA.elements[idx2d], 2);
    r += vp ;
   
    //printf("\n\nRow(%u), Col(%u), Matrix.width(%u), Matrix.height(%u), idx2d(%u), mA.elements(%f), pow(mA.elements[idx2d](%f) mxn(%u) sum(%f)", row, col, mA.width, mA.height, idx2d, mA.elements[idx2d], vp, mxn, r);
   }

   unsigned int idxvector = row * mR.width + col;
   mR.elements[idxvector] = r;
   //to debug indexes and values
   //printf("\n Row(%u), Col(%u), Matrix.width(%u), Matrix.height(%u), IdxVector(%u), sum(%f) mR.elements[idxvector](%f)", row, col, mA.width, mA.height, idxvector, r, mR.elements[idxvector]);
  } 
}


/*
Kernel implementation for power calculation on each element in the matrix
It receives a matrix mA and update each element with the power value, assigning each thread to each element
*/
__global__ void power_matrix_kernel_cuda_struct(const Matrix mA, const Matrix mR){

  //for a scenario power_matrix_kernel_cuda_struct<<<100, 25>>>(...);
     //inside the kernel, each thread can calculate a unique id with:
	//int id = blockid.x * blockDim.x + threadIdx.x;
    //the 14th thread of the 76th block would calculate:
	//int id = 76 * 25 + 14 //array element index 1914
    
   /*
   Matrix sizes to test: 2000x4000, 4000x2000, 10000x40000, 40000x10000
   Threads per block: 32, 64, 128
   */

    //2D Thread ID
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;
    int mn = mA.width * mA.height;
    int row = (blockIdx.x * blockDim.x + threadIdx.x)/mn;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)%mn;

    unsigned int idx = row * mA.height + col;

  //printf("\n idx:%u original value:%f m x n:%f", idx, mA.elements[idx], mxn);

  //to ensure threads will never point outside matrix boundaries
  if (idx < mn){
    float v = mA.elements[idx];
    mR.elements[idx] =  v * v;
    //printf("\n idx:%u New MatrixElement:%f original value:%f", idx, mA.elements[idx], v);
  }
}

/*
Kernel implementation for reduce sum on each row from powered Matrix
It receives a matrix mA and assignd each thread to each row to perform the reduce sum and store the result in th vector mR
*/
__global__ void reduce_sum_vector_kernel_cuda_struct(const Matrix mA,const Matrix mR){
    //unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    int mxn = mA.width * mA.height;
    int row = (blockIdx.x * blockDim.x + threadIdx.x)/mxn;
    int col = (blockIdx.x * blockDim.x + threadIdx.x)%mxn;

    unsigned int idx = row * mA.width + col;

    float r = 0;
    
    //to ensure no thread will point outside matrix boundary (height)
    if(idx < mA.height){
      //to reduce as a summarize operation on elements of a row from the power matrix
     for(unsigned int j = 0; j < mA.width; j++){
      unsigned int idx2d = col * mA.width + j;
      r += mA.elements[idx2d];
      
      //printf("\n\nRow(%u), Col(%u), Matrix.width(%u), Matrix.height(%u), idx2d(%u), mA.elements(%f) mxn(%u)", row, col, mA.width, mA.height, idx2d, mA.elements[idx2d], mxn);
     } 

     unsigned int idxvector = row * mR.width + col;
     mR.elements[idxvector] = r;
     //to debug indexes and values
     //printf("\n Row(%u), Col(%u), Matrix.width(%u), Matrix.height(%u), IdxVector(%u), sum(%f)", row, col, mA.width, mA.height, idxvector, r);
    }
}

/*
  Sacar por pantalla información del *device*
*/
void devicenfo(void)
{ 
  struct cudaDeviceProp capabilities;
  
  cudaGetDeviceProperties (&capabilities, 0);
  
  printf("->CUDA Platform & Capabilities\n");
  printf("Name: %s\n", capabilities.name);
  printf("totalGlobalMem: %.2f MB\n", capabilities.totalGlobalMem/1024.0f/1024.0f);
  printf("sharedMemPerBlock: %.2f KB\n", capabilities.sharedMemPerBlock/1024.0f);
  printf("regsPerBlock (32 bits): %d\n", capabilities.regsPerBlock);
  printf("warpSize: %d\n", capabilities.warpSize);
  printf("memPitch: %.2f KB\n", capabilities.memPitch/1024.0f);
  printf("maxThreadsPerBlock: %d\n", capabilities.maxThreadsPerBlock);
  printf("maxThreadsDim: %d x %d x %d\n", capabilities.maxThreadsDim[0],
         capabilities.maxThreadsDim[1], capabilities.maxThreadsDim[2]);
  printf("maxGridSize: %d x %d\n", capabilities.maxGridSize[0],
         capabilities.maxGridSize[1]);
  printf("totalConstMem: %.2f KB\n", capabilities.totalConstMem/1024.0f);
  printf("major.minor: %d.%d\n", capabilities.major, capabilities.minor);
  printf("clockRate: %.2f MHz\n", capabilities.clockRate/1024.0f);
  printf("deviceOverlap: %d\n", capabilities.deviceOverlap);
  printf("multiProcessorCount: %d\n", capabilities.multiProcessorCount);
}


/*
  Función que ajusta el número de hilos, de bloques, y de bloques por hilo 
  de acuerdo a las restricciones de la GPU
*/
void checkparams(unsigned int *n, unsigned int *m, unsigned int *cb)
{
  struct cudaDeviceProp capabilities;

  //Total number of elements in the matrix
  unsigned int nm = *n * *m;

  // If there are more threads than elements in the matrix, the latter value becomes the block size
  if (*cb > nm)
    *cb = nm;

  cudaGetDeviceProperties (&capabilities, 0);

  if (*cb > capabilities.maxThreadsDim[0]) {
    *cb = capabilities.maxThreadsDim[0];
    printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n\n",
           *cb);
  }

  if (((nm + *cb - 1) / *cb) > capabilities.maxGridSize[0]) {
    *cb = 2 * (nm - 1) / (capabilities.maxGridSize[0] - 1);
    if (*cb > capabilities.maxThreadsDim[0]) {
      *cb = capabilities.maxThreadsDim[0];
      printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n", *cb);
      if (nm > (capabilities.maxGridSize[0] * *cb)) {
        nm = capabilities.maxGridSize[0] * *cb;
        printf("->Núm. total de hilos cambiado a %d (máx por grid para dev)\n\n", nm);
      } else {
        printf("\n");
      }
    } else {
      printf("->Núm. hilos/bloq cambiado a %d (%d máx. bloq/grid para dev)\n\n",
             *cb, capabilities.maxGridSize[0]);
    }
  }
}
