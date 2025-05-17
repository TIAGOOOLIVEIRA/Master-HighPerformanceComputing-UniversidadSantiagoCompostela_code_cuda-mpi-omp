

multf 
compute -c 4

gcc -O2 -fno-tree-vectorize -fopenmp -fopt-info-vec -o multf multf.c
single thread execution - Commented OpenMP declaratives
./multf 
 Execution time = 12.0903 seconds
 Execution time = 12.1125 seconds
 Execution time = 12.12 seconds



***gcc -O2 -fopenmp -fopt-info-vec -march=native
***gcc-14 -fopenmp -O3 -march=native -o multf multf.c

export OMP_NUM_THREADS=4
gcc -O2 -fno-tree-vectorize -fopenmp -fopt-info-vec -o multf multf.c
parallelism - enabled OpenMP declaratives, without vectorization
./multf
 Execution time = 3.0328 seconds
 Execution time = 3.0332 seconds
 Execution time = 3.03379 seconds

export OMP_NUM_THREADS=2
 Execution time = 6.07174 seconds
 Execution time = 6.07249 seconds
 Execution time = 6.07154 seconds

export OMP_NUM_THREADS=1
 Execution time = 12.1577 seconds
 Execution time = 12.1587 seconds
 Execution time = 12.1558 seconds


vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./multf
...
Vectorization: 0.0% of Packed FP Operations
 | This code has floating point operations and is not vectorized. Consider
 | either recompiling the code with optimization options that allow
 | vectorization or using Intel Advisor to vectorize the loops.
 |
    Instruction Mix
        SP FLOPs: 39.9% of uOps
            Packed: 0.0% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 0.0% from SP FP
            Scalar: 100.0% from SP FP
             | This code has floating point operations and is not vectorized.
             | Consider either recompiling the code with optimization options
             | that allow vectorization or using Intel Advisor to vectorize the
             | loops.
             |
        DP FLOPs: 0.0% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 0.0% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 60.1% of uOps
    FP Arith/Mem Rd Instr. Ratio: 0.997
    FP Arith/Mem Wr Instr. Ratio: 4,084.218
...

export OMP_NUM_THREADS=4
gcc -fopenmp -O3 -march=native -ftree-vectorize -fopt-info-vec -o multf_vec multf_vec.c
multf_vec.c:45:18: optimized: loop vectorized using 64 byte vectors
multf_vec.c:45:18: optimized: loop vectorized using 32 byte vectors
multf_vec.c:47:24: optimized: loop vectorized using 64 byte vectors
multf_vec.c:29:7: optimized: loop vectorized using 64 byte vectors
multf_vec.c:26:7: optimized: loop vectorized using 64 byte vectors
./multf_vec
 Execution time = 0.311219 seconds
 Execution time = 0.311368 seconds
 Execution time = 0.311328 seconds

export OMP_NUM_THREADS=2
 Execution time = 0.583954 seconds
 Execution time = 0.586333 seconds
 Execution time = 0.609152 seconds

export OMP_NUM_THREADS=1
 Execution time = 1.04943 seconds
 Execution time = 1.05038 seconds
 Execution time = 1.0497 seconds


vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./multf_vec
...
Vectorization: 94.0% of Packed FP Operations
    Instruction Mix
        SP FLOPs: 32.2% of uOps
            Packed: 94.0% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 94.0% from SP FP
            Scalar: 6.0% from SP FP
        DP FLOPs: 0.0% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 0.0% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 67.8% of uOps
    FP Arith/Mem Rd Instr. Ratio: 1.014
    FP Arith/Mem Wr Instr. Ratio: 2.012
...

likwid