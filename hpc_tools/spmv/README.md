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

________________________________________________________________________________________________________
Future Work: Profiling (to spot optimization made on the code by the compiler):
Likwid:
Gprof:
Valgrind:
VTune:
Perf:

Memory of work:

$module load openblas

    -O0
    gcc -O0 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc  Ref
#1  768         131     105     121     74
#2  767         131     105     120     74
#3  769         131     104     120     75
avg

    -O2 -fno-tree-vectorize
    gcc -O2 -fno-tree-vectorize -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc  Ref
#1  361         83      35      30      75
#2  358         74      35      37      83
#3  361         73      35      35      74
avg

    -O3
    gcc -O3 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc  Ref
#1  364         74      34      33      74
#2  371         75      35      34      78
#3  361         74      39      30      74
avg

    -Ofast
    gcc -Ofast -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    My_Dense    My_coo  My_csr  My_csc  Ref
#1  359         74      40      31      74
#2  371         75      39      32      75
#3  367         73      34      30      74

avg
