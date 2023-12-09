
He creado las siguientes funciones abajo siguiendo una estrategia de dividir el trabajo en dos tareas principales y ejecutarlas como un pipeline: Una función, responsable de calcular la potencia al cuadrado de cada elemento en la matriz en paralelo dentro del dispositivo GPU; otra, encargada de recibir la matriz del trabajo anterior, asignando cada fila de la matriz a un hilo del dispositivo GPU para que puedan ejecutar la suma de sus elementos en paralelo y almacenar este valor en el vector de resultado final.


    1. fill_matrix: Responsibe to create a 2D matrix filled with int ascending values starting from 1
        adding +1 to upcoming position until n size of the dimension 
    2. power_matrix_kernel_cuda: Responsible to calculate the power value for each element in the matrix on GPU
    3. reduce_sum_vector_kernel_cuda: Responsible to summarize each row in the matrizx and return a vector with
        the reduced value per row on GPU
    4. power_matrix_GPU: responsible to manage memory de/allocation for CPU/GPU, block/grid definition and 
        profiling GPU execution as well for power computing
    5. sum_vector_GPU: responsible to manage memory de/allocation for CPU/GPU, block/grid definition and
        profiling GPU execution as well for sum vector elements computing


Consideraciones: Debido a la falta de experiencia en el manejo de matrices en lenguaje C, tuve dificultades para pasarlas como parámetros en las funciones de CPU y GPU.
Por tanto, la aplicación sigue teniendo fallos sobre este tema. Sin embargo, estoy seguro de que todo el flujo descrito anteriormente es coherente.

Reconozco que la tarea está incompleta, por lo tanto, solicito amablemente más tiempo para poder corregir estas secciones de paso de parámetros y así ejecutar la aplicación y recopilar sus estadísticas dependiendo de los escenarios de prueba propuestos.

Error que aparece al compilar:
 error: expression must have pointer-to-object type
 


compute --gpu
nvcc -o eucvectornorm assingment1_eucvectornorm.cu
sbatch job.sh
watch -n 1 squeue -u curso370