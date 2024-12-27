# HPC_hpctools_spmv
• Ok, you already have your own CSR and dense SpMV functions. Now, code your COO (triplet) and your CSC versions of SpMV, and run a benchmark with all 4 implementations.

• Fill out two tables with the measured execution times obtained for your 4 kernels when working with 16384 × 16384 matrices with a 10% non-zero elements, compiled with different optimizations (see next slide).

• In the first table, GCC is used for the experiments, while ICC binaries are employed on the second table.

• For each compiler, you will compare 4 different versions of your functions, applying different compiler optimizations (according to the column names in the tables): no optimization at all, O2 without vectorization, O3 with autovectorization, and Ofast with autovectorization (use -fast in ICC instead).
• The last column, Ref, will show the reference time for each operation, using GSL in Table 1 and Intel MKL in Table 2.

Table 1 gcc Benchmark - 16384 x 16384 matrices, 10% non-zero elements
            O0  02-novec    03-vec  Ofast-vec   Ref
My_Dense
My_coo
My_csr
My_csc

Table 1 icc Benchmark - 16384 x 16384 matrices, 10% non-zero elements
            O0  02-novec    03-vec  Ofast-vec   Ref
My_Dense
My_coo
My_csr
My_csc


Profiling (to spot optimization made on the code by the compiler):
Likwid:
Gprof:
Valgrind:
VTune:
Perf:




    -O0
    gcc -O0 -lopenblas -lgslcblas spmv.c timer.c my_dense.c my_csr.c -o spmv
    My_Dense    My_coo  My_csr  My_csc
#1 835                  257
#2 841                  256
#3 850                  257
#4 841                  280
#5 883                  261
avg