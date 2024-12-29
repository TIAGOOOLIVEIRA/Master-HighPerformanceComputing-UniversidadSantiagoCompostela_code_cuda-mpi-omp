# HPC Tool - HPC Tools block 2 - Task 2: SpMV benchmarking

• Ok, you already have your own CSR and dense SpMV functions. Now, code your COO (triplet) and your CSC versions of SpMV, and run a benchmark with all 4 implementations.

• Fill out two tables with the measured execution times obtained for your 4 kernels when working with 16384 × 16384 matrices with a 10% non-zero elements, compiled with different optimizations (see next slide).

• In the first table, GCC is used for the experiments, while ICC binaries are employed on the second table.

• For each compiler, you will compare 4 different versions of your functions, applying different compiler optimizations (according to the column names in the tables): no optimization at all, O2 without vectorization, O3 with autovectorization, and Ofast with autovectorization (use -fast in ICC instead).
• The last column, Ref, will show the reference time for each operation, using GSL in Table 1 and Intel MKL in Table 2.

Table 1 gcc Benchmark - 16384 x 16384 matrices, 10% non-zero elements
| SpMV-func   | O0    | 02-novec | 03-vec | Ofast-vec |
|-------------|--------|----------|--------|-----------|
| My_Dense    | 768    | 360      | 365.3  | 365.7     |
| My_coo      | 131    | 76.7     | 74.3   | 74        |
| My_csr      | 104.7  | 35       | 36     | 37.7      |
| My_csc      | 120.3  | 34       | 32.3   | 31        |
| Ref         | 74.3   | 77.3     | 75.3   | 74.3      |


Table 1 icc Benchmark - 16384 x 16384 matrices, 10% non-zero elements
| SpMV-func   | O0    | 02-novec | 03-vec | Ofast-vec |
|-------------|--------|----------|--------|-----------|
| My_Dense    | 766    | 337.7    | 150    | 149.7     |
| My_coo      | 135.7  | 72       | 72     | 71.7      |
| My_csr      | 99.7   | 37       | 22.7   | 22        |
| My_csc      | 110.7  | 28       | 27.7   | 28.3      |
| Ref         |        |          |        |           |


## Memory of work:
- GCC 

$module load openblas


    O0
    gcc -O0 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 768      | 131    | 105    | 121    | 74    |
| #2    | 767      | 131    | 105    | 120    | 74    |
| #3    | 769      | 131    | 104    | 120    | 75    |
| **avg** | 768      | 131    | 104.7  | 120.3  | 74.3  |


    O2 -fno-tree-vectorize
    gcc -O2 -fno-tree-vectorize -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 361      | 83     | 35     | 30     | 75    |
| #2    | 358      | 74     | 35     | 37     | 83    |
| #3    | 361      | 73     | 35     | 35     | 74    |
| **avg** | 360      | 76.7   | 35     | 34     | 77.3  |


    O3
    gcc -O3 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 364      | 74     | 34     | 33     | 74    |
| #2    | 371      | 75     | 35     | 34     | 78    |
| #3    | 361      | 74     | 39     | 30     | 74    |
| **avg** | 365.3    | 74.3   | 36     | 32.3   | 75.3  |


    Ofast
    gcc -Ofast -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 359      | 74     | 40     | 31     | 74    |
| #2    | 371      | 75     | 39     | 32     | 75    |
| #3    | 367      | 73     | 34     | 30     | 74    |
| **avg** | 365.7    | 74     | 37.7   | 31     | 74.3  |


- ICC 

$module load intel


    O0
    icc -O0 spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 760      | 130    | 98     | 109    |       |
| #2    | 766      | 144    | 100    | 112    |       |
| #3    | 772      | 133    | 101    | 111    |       |
| **avg** | 766      | 135.7  | 99.7  | 110.7  |       |


    O2 -fno-tree-vectorize
    icc -O2 -fno-tree-vectorize spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 338      | 72     | 44     | 28     |       |
| #2    | 337      | 72     | 33     | 28     |       |
| #3    | 338      | 72     | 34     | 28     |       |
| **avg** | 337.7    | 72     | 37     | 28     |       |


    O3
    icc -O3 spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 151      | 72     | 24     | 28     |       |
| #2    | 149      | 72     | 22     | 27     |       |
| #3    | 150      | 72     | 22     | 28     |       |
| **avg** | 150      | 72     | 22.7   | 27.7   |       |


    Ofast
    icc -Ofast spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 150      | 72     | 22     | 28     |       |
| #2    | 148      | 72     | 22     | 29     |       |
| #3    | 151      | 71     | 22     | 28     |       |
| **avg** | 149.7    | 71.7   | 22     | 28.3   |       |

________________________________________________________________________________________________________
- Issues when loading mkl: *mkl not found

$find / -name "libmkl_core.so" 2>/dev/null

  - icc -O0 spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lgsl -lgslcblas -lpthread -lm -ldl -o spmv
  - /mnt/netapp1/Optcesga_FT2_RHEL7/2020/gentoo/22072020/usr/lib/gcc/x86_64-pc-linux-gnu/10.1.0/../../../../x86_64-pc-linux-gnu/bin/ld: cannot find -lmkl_intel_lp64
  - /mnt/netapp1/Optcesga_FT2_RHEL7/2020/gentoo/22072020/usr/lib/gcc/x86_64-pc-linux-gnu/10.1.0/../../../../x86_64-pc-linux-gnu/bin/ld: cannot find -lmkl_core
  - /mnt/netapp1/Optcesga_FT2_RHEL7/2020/gentoo/22072020/usr/lib/gcc/x86_64-pc-linux-gnu/10.1.0/../../../../x86_64-pc-linux-gnu/bin/ld: cannot find -lmkl_sequential



________________________________________________________________________________________________________
## Future Work

- Profiling (to spot optimization made on the code by the compiler):
  - Likwid
  - Gprof
  - Valgrind
  - VTune
  - Perf
