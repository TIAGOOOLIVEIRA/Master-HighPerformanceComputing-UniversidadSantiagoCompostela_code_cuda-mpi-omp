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



$module load openblas

    -O0
    gcc -O0 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc
#1 835          324     257     369
#2 841          324     256     291
#3 850          375     257     290
#4 841          325     280     311
#5 883          326     261     371
avg

    -O2 -fno-tree-vectorize
    gcc -O2 -fno-tree-vectorize -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc
#1  448         185     89      79
#2  480         185     116     83
#3  493         185     89      77
#4  496         185     90      82
#5  474         186     96      86
avg

    -O3
    gcc -O3 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc
#1  358         186     91      82
#2  358         185     90      77
#3  362         185     90      78
#4  356         184     90      80
#5  361         185     90      81
avg

    -Ofast
    gcc -Ofast -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc
#1  361         185     91      75
#2  368         185     91      73
#3  358         185     89      73
#4  361         185     91      71
#5  360         184     91      68
avg
