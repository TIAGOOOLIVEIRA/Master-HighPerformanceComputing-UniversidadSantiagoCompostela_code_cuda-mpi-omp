# OpenMP and MPI Labs Assignment

## Labs1, Vectorization with OpenMP; 1. The code multf.c performs the product of matrices

Performance & Speedup Analysis: Matrix Multiplication (D = A × Bᵗ)

- compute -c 4
- module load intel vtune
- export OMP_NUM_THREADS={1,2,4}

Four versions of the matrix multiplication application were evaluated:

- **Baseline**: Sequential version (OpenMP directives commented)
- **OMP**: OpenMP parallelized version (no vectorization)
- **OMP + SIMD**: OpenMP parallelized and SIMD for omp reduction
- **OMP + SIMD + flags**: OpenMP parallelized, SIMD for omp reduction and agressive optimizations with compiler-enabled SIMD auto-vectorization 

### Compiler and flags used:

- **Makefile**: Make
  - **make MODE=sequential**

        Baseline Sequential
        
  - **make MODE=parallel**
        
        OpenMP Parallel

  - **make TARGET=multf_vec MODE=simd**
        
        SIMD Declaratives (OpenMP + SIMD code)

  - **make TARGET=multf_vec MODE=vector**

        SIMD + Autovectorization Flags

  - **make clean**
        
        Clean up binaries


- **Manual compilation (Linux)**: 
```bash
# Baseline
gcc -O2 -fno-tree-vectorize -fopenmp -o multf multf.c

# Parallel (OpenMP)
gcc -O2 -fopenmp -fopt-info-vec -march=native -o multf multf.c

# Parallel OpenMP with SIMD code declaratives
gcc -O2 -fopenmp -fopt-info-vec -march=native -o multf_vec multf_vec.c
multf_vec.c:47:24: optimized: loop vectorized using 64 byte vectors

# Vectorized (OpenMP + SIMD + Autovectorizing flags)
gcc -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec -o multf_vec multf_vec.c
multf_vec.c:45:18: optimized: loop vectorized using 64 byte vectors
multf_vec.c:45:18: optimized: loop vectorized using 32 byte vectors
multf_vec.c:47:24: optimized: loop vectorized using 64 byte vectors
multf_vec.c:29:7: optimized: loop vectorized using 64 byte vectors
multf_vec.c:26:7: optimized: loop vectorized using 64 byte vectors



```

### Average Execution Time (in seconds)
The applications were executed three times to get the average value for a better statistical relevance.

| Version              | Threads | Avg Time (s) |
|----------------------|---------|--------------|
| **Baseline**         | 1       | 12.1076      |
| **OMP (No SIMD)**    | 1       | 12.1574      |
|                      | 2       | 6.0719       |
|                      | 4       | 3.0333       |
| **OMP + SIMD**       | 1       | 2.4343       |
|                      | 2       | 1.1901       |
|                      | 4       | 0.6352       |
| **OMP + SIMD + flags**       | 1       | 1.0498       |
|                      | 2       | 0.5931       |
|                      | 4       | 0.3113       |


### Speedup Relative to Baseline
| Version              | Threads | Speedup |
|----------------------|---------|---------|
| **Baseline**         | 1       | 1.00×    |
| **OMP (No SIMD)**    | 1       | 0.9967×  |
|                      | 2       | 1.9933×  |
|                      | 4       | 3.9906×  |
| **OMP + SIMD**       | 1       | 4.97×    |
|                      | 2       | 10.17×   |
|                      | 4       | 19.06×   |
| **OMP + SIMD + flags**       | 1       | 11.53×   |
|                      | 2       | 20.41×   |
|                      | 4       | 38.89×   |



 ### Vectorization Insights (Intel VTune Analysis)
| Version       | Vectorized FP Ops | SIMD Width Used  |
|---------------|-------------------|------------------|
| **OMP Only**  | 0.0%              | Scalar only      |
| **OMP + SIMD**| 94.1%             | 512-bit (AVX-512)|
| **OMP + SIMD + flags**| 94.0%             | 512-bit (AVX-512)|

- **Vectorization report - OMP Only - VTune**: For the parallelized with OpenMP, without SIMD vectorization
```bash
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
```

- **Vectorization report - OMP + SIMD - VTune**: For the parallelized with OpenMP, with SIMD vectorization
```bash
vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./multf_vec
...
Vectorization: 94.1% of Packed FP Operations
    Instruction Mix
        SP FLOPs: 27.7% of uOps
            Packed: 94.1% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 94.1% from SP FP
            Scalar: 5.9% from SP FP
        DP FLOPs: 0.0% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 0.0% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 72.3% of uOps
    FP Arith/Mem Rd Instr. Ratio: 0.678
    FP Arith/Mem Wr Instr. Ratio: 2.056
...
```


- **Vectorization report - OMP + SIMD + flags - VTune**: For the parallelized with OpenMP, with SIMD vectorization and compiler flags for auto-vectorization
```bash
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

```

- **[WiP]Vectorization report - OMP Only - Likwid**: For the parallelized with OpenMP, without SIMD vectorization


- **[WiP]Vectorization report - OMP + SIMD - Likwid**: For the parallelized with OpenMP, with SIMD vectorization

- **[WiP]Vectorization report - OMP + SIMD + flags - Likwid**: For the parallelized with OpenMP, with SIMD vectorization

### Future work

To further improve vectorization, memory access efficiency, and profiling, the following GCC flags are commonly used:



| Flag                          | Purpose                                                                                   |
|------------------------------|--------------------------------------------------------------------------------------------|
| `-O3`                         | Enables high-level optimizations including loop unrolling and inlining.                   |
| `-Ofast`                      | Similar to `-O3` but also ignores strict standards compliance for faster code.            |
| `-march=native`              | Generates code optimized for the architecture of the host machine.                        |
| `-mtune=native`              | Tunes performance (scheduling, pipelining) for the host CPU without changing ISA.         |
| `-mprefer-vector-width=512` | Instructs compiler to prefer 512-bit SIMD width (for AVX-512 CPUs).                       |
| `-fopenmp`                   | Enables OpenMP multithreading.                                                            |
| `-ftree-vectorize`           | Enables automatic loop vectorization.                                                     |
| `-ftree-loop-vectorize`      | More explicit flag to vectorize loops (redundant with `-ftree-vectorize`, but clearer).   |
| `-ftree-slp-vectorize`       | Enables basic block-level vectorization (Superword Level Parallelism).                    |
| `-fstrict-aliasing`          | Assumes that pointer aliases follow standard rules, allowing better vectorization.       |
| `-fno-math-errno`            | Avoids storing `errno` for math functions, enabling math inlining/vectorization.         |
| `-fopt-info-vec`             | Reports which loops were vectorized.                                                      |
| `-fopt-info-vec-missed`      | Reports which loops **were not** vectorized and why.                                     |
| `-fopt-info-vec-all`         | Shows all vectorization attempts (successful or not).                                     |
| `-fopt-info-optall-optimized`| Reports **all optimizations** made by the compiler.                                       |
| `-fdump-tree-vect`           | Dumps detailed vectorization info (in `.vect` dump files, e.g., `multf.c.147t.vect`).    |

---

- **The `restrict` keyword**: tells the compiler that a given pointer is the **only** reference to that memory region during its scope. This guarantees **no aliasing**, which allows:

  - Better **vectorization** because the compiler can confidently reorder or combine memory loads/stores.
  - More aggressive optimizations because pointer aliasing checks are unnecessary.


### Conclusions
- OpenMP-only version scales nearly linearly with threads (up to 4×), but misses deeper compiler optimizations.
- Adding compiler vectorization flags unlocks full hardware performance as per the aggressive compiler optimizations and loop transformations (-O3, -ftree-vectorize, -march=native): 
  - significantly outperforms others, achieving ~39× speedup;
  - Maximizes SIMD utilization (as seen in VTune reports); 
  - Reduces scalar overhead through inlining, unrolling, and optimized memory access.
- VTune and Likwid confirms 94% vectorized FP instructions, leveraging AVX-512 (64-byte SIMD).
- Combining OpenMP for multithreading with SIMD via vectorization maximizes performance in CPU-bound workload.


## Labs1, Vectorization with OpenMP; 2,3. Parallelize and vectorize saxpy.c  

- compute -c 4
- module load intel vtune
- export OMP_NUM_THREADS={1,2,4}

Three versions of the saxpy factor multiplication application were evaluated:

- **SIMD only**: Only SIMD function declaration
- **OMP + SIMD**: OpenMP loop parallelized and SIMD function declaration
- **OMP + SIMD + flags**: OpenMP parallelized, SIMD function declaration and agressive optimization with compiler-enabled SIMD auto-vectorization 


### Compiler and flags used:

- **Makefile**: Make
  - **make MODE=basic**

        Basic mode

  - **make MODE=report**

        With vectorization reporting

  - **make MODE=vectorized**

        Fully optimized (O3 + vectorization)

  - **make clean**

        Clean build                        

- **Manual compilation (Linux)**: 
```bash
gcc -O2 -fopenmp -fopt-info-vec -o saxpy saxpy.c

gcc -O2 -fopenmp -fopt-info-vec -fopt-info-vec-optimized -march=native -o saxpy saxpy.c
saxpy.c:108:14: optimized: loop vectorized using 64 byte vectors
saxpy.c:16:15: optimized: loop vectorized using 64 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 16 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 32 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 32 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 64 byte vectors

gcc -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec -o saxpy saxpy.c
saxpy.c:74:5: optimized: loop vectorized using 64 byte vectors
saxpy.c:74:5: optimized: loop vectorized using 64 byte vectors
saxpy.c:16:15: optimized: loop vectorized using 64 byte vectors
saxpy.c:16:15: optimized: loop vectorized using 64 byte vectors
saxpy.c:99:5: optimized: loop vectorized using 64 byte vectors
saxpy.c:99:5: optimized: loop vectorized using 64 byte vectors
saxpy.c:112:14: optimized: loop vectorized using 64 byte vectors
saxpy.c:112:14: optimized: loop vectorized using 64 byte vectors
saxpy.c:51:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:61:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:61:3: optimized: loop vectorized using 32 byte vectors
saxpy.c:61:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:51:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:61:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:51:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:61:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:51:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:61:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:51:3: optimized: loop vectorized using 64 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 16 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 32 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 32 byte vectors
saxpy.c:26:12: optimized: loop vectorized using 64 byte vectors
```

### SIMD only Baseline Execution

The applications were executed three times to get the average value for a better statistical relevance.

| Function        | Baseline Avg Time (s) |
|----------------|------------------------|
| saxpy_no_simd  | 1.086                  |
| saxpy          | 0.937                  |
| saxpyi_no_simd | 1.304                  |
| saxpyi         | 0.938                  |


### SIMD + OpenMP Loops Parallelized 

| Function        | Threads | Run 1  | Run 2  | Run 3  | **Average Time (s)** |
|----------------|---------|--------|--------|--------|-----------------------|
| saxpy_no_simd  | 4       | 0.281  | 0.282  | 0.278  | **0.280**             |
| saxpy          | 4       | 0.210  | 0.208  | 0.208  | **0.209**             |
| saxpyi_no_simd | 4       | 0.362  | 0.363  | 0.359  | **0.361**             |
| saxpyi         | 4       | 0.210  | 0.209  | 0.208  | **0.209**             |
| saxpy_no_simd  | 2       | 0.576  | 0.560  | 0.558  | **0.565**             |
| saxpy          | 2       | 0.437  | 0.415  | 0.425  | **0.426**             |
| saxpyi_no_simd | 2       | 0.732  | 0.721  | 0.718  | **0.724**             |
| saxpyi         | 2       | 0.438  | 0.416  | 0.425  | **0.426**             |
| saxpy_no_simd  | 1       | 1.087  | 1.088  | 1.084  | **1.086**             |
| saxpy          | 1       | 0.942  | 0.939  | 0.931  | **0.937**             |
| saxpyi_no_simd | 1       | 1.304  | 1.305  | 1.302  | **1.304**             |
| saxpyi         | 1       | 0.943  | 0.939  | 0.932  | **0.938**             |


### SIMD + OpenMP Loops Parallelized - Summary (OpenMP + SIMD, 4 Threads)

| Function        | Optimized Avg Time (s) |
|----------------|-------------------------|
| saxpy_no_simd  | 0.280                   |
| saxpy          | 0.209                   |
| saxpyi_no_simd | 0.361                   |
| saxpyi         | 0.209                   |


### Speedup: SIMD Only vs Optimized (OpenMP + SIMD, 4 Threads)

| Function        | Baseline (s) | Optimized (s) | Speedup |
|----------------|--------------|----------------|---------|
| saxpy_no_simd  | 1.086        | 0.280          | 3.88×   |
| saxpy          | 0.937        | 0.209          | 4.48×   |
| saxpyi_no_simd | 1.304        | 0.361          | 3.61×   |
| saxpyi         | 0.938        | 0.209          | 4.49×   |

----

### SIMD + OpenMP Loops Parallelized + flags for auto-vectorization

| Function        | Threads | Run 1  | Run 2  | Run 3  | Avg (s) |
|----------------|---------|--------|--------|--------|---------|
| saxpy_no_simd  | 4       | 0.106  | 0.107  | 0.108  | 0.107   |
| saxpy          | 4       | 0.106  | 0.107  | 0.108  | 0.107   |
| saxpyi_no_simd | 4       | 0.106  | 0.107  | 0.108  | 0.107   |
| saxpyi         | 4       | 0.106  | 0.107  | 0.108  | 0.107   |
| saxpy_no_simd  | 2       | 0.211  | 0.213  | 0.210  | 0.211   |
| saxpy          | 2       | 0.211  | 0.213  | 0.209  | 0.211   |
| saxpyi_no_simd | 2       | 0.211  | 0.214  | 0.209  | 0.211   |
| saxpyi         | 2       | 0.210  | 0.213  | 0.210  | 0.211   |
| saxpy_no_simd  | 1       | 0.483  | 0.482  | 0.481  | 0.482   |
| saxpy          | 1       | 0.483  | 0.481  | 0.480  | 0.481   |
| saxpyi_no_simd | 1       | 0.483  | 0.482  | 0.480  | 0.482   |
| saxpyi         | 1       | 0.483  | 0.482  | 0.480  | 0.482   |


### Speedup SIMD Only vs SIMD + OpenMP Loops Parallelized + flags for auto-vectorization

| Function        | Threads | Baseline (s) | Optimized (s) | Speedup |
|----------------|---------|---------------|----------------|---------|
| saxpy_no_simd  | 4       | 1.086         | 0.107          | 10.15×  |
| saxpy          | 4       | 0.937         | 0.107          | 8.76×   |
| saxpyi_no_simd | 4       | 1.304         | 0.107          | 12.18×  |
| saxpyi         | 4       | 0.938         | 0.107          | 8.77×   |
| saxpy_no_simd  | 2       | 1.086         | 0.211          | 5.15×   |
| saxpy          | 2       | 0.937         | 0.211          | 4.44×   |
| saxpyi_no_simd | 2       | 1.304         | 0.211          | 6.18×   |
| saxpyi         | 2       | 0.938         | 0.211          | 4.44×   |
| saxpy_no_simd  | 1       | 1.086         | 0.482          | 2.25×   |
| saxpy          | 1       | 0.937         | 0.481          | 1.95×   |
| saxpyi_no_simd | 1       | 1.304         | 0.482          | 2.71×   |
| saxpyi         | 1       | 0.938         | 0.482          | 1.95×   |



### Vectorization Insights (Intel VTune Analysis)

- **Vectorization report - SIMD + OpenMP - VTune**: For the loops parallelized with OpenMP and SIMD function vectorization
```bash
vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./saxpy
...
Vectorization: 19.9% of Packed FP Operations
 | A significant fraction of floating point arithmetic instructions are scalar.
 | This indicates that the code was not fully vectorized. Use Intel Advisor to
 | see possible reasons why the code was not vectorized.
 |
    Instruction Mix
        SP FLOPs: 27.6% of uOps
            Packed: 19.9% from SP FP
                128-bit: 19.9% from SP FP
                 | Using the latest vector instruction set can improve
                 | parallelism for this code. Consider either recompiling the
                 | code with the latest instruction set or using Intel Advisor
                 | to get vectorization help.
                 |
                256-bit: 0.0% from SP FP
                512-bit: 0.0% from SP FP
            Scalar: 80.1% from SP FP
             | A significant fraction of floating point arithmetic instructions
             | are scalar. This indicates that the code was not fully
             | vectorized. Use Intel Advisor to see possible reasons why the
             | code was not vectorized.
             |
        DP FLOPs: 0.0% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 0.0% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 72.4% of uOps
    FP Arith/Mem Rd Instr. Ratio: 0.981
    FP Arith/Mem Wr Instr. Ratio: 1.999
...
```


- **Vectorization report - SIMD + OpenMP + flags auto vectorizing - VTune**: For the parallelized with OpenMP, with SIMD vectorization and compiler flags for auto-vectorization
```bash
vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./saxpy
...
Vectorization: 87.2% of Packed FP Operations
    Instruction Mix
        SP FLOPs: 51.6% of uOps
            Packed: 87.2% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 87.2% from SP FP
            Scalar: 12.8% from SP FP
        DP FLOPs: 0.0% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 0.0% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 48.4% of uOps
    FP Arith/Mem Rd Instr. Ratio: 1.444
    FP Arith/Mem Wr Instr. Ratio: 4.046
...
```

### VTune comparison: SIMD + OpenMP vs SIMD + OpenMP + 03, Auto vectorozing

| Aspect                 | No `-O3`, No Auto Vec      | With `-O3` and Auto Vec      |
|------------------------|----------------------------|------------------------------|
| Vectorization (%)      | 19.9%                      | 87.2%                        |
| SIMD Width             | 128-bit (SSE)              | 512-bit (AVX-512)            |
| Scalar FP              | 80.1%                      | 12.8%                        |
| Performance Potential  | Poor (mostly scalar loops) | Excellent (wide SIMD usage)  |
| Compiler Strategy      | Conservative               | Aggressive + intelligent     |

### Conclusions
- All four function variants experienced 3.6× to 4.5× speedup when parallelism and SIMD function vectorization were enabled. Having best observed speedup:
  - saxpyi: 4.49×
  - saxpy: 4.48×
- Combining OpenMP threading with SIMD vectorization is crucial for exploiting modern CPU hardware effectively
- Autovectorization + OpenMP results in a massive boost when loops are structured well
- It would worth to redesign the loops for leveraging omp [colapse (2)] so the product of the iteration spaces (NREPS * N) becomes the total loop space.


## Labs1, Vectorization with OpenMP; 4. Vectorize & parallelize the code jacobi.c



- compute -c 4
- module load intel vtune
- export OMP_NUM_THREADS={1,4,8}

### Compiler and flags used:


- **Makefile**: Make/Gprof
  - **make MODE=profile**

        Basic mode

  - **make MODE=vectorized**

        To add auto-vectorizing flag to compiler

  - **make run-profile**

        Runs program to generate gmon.out

  - **make gprof-report**

        Analyzes and dumps result

  - **cat gprof-report.txt**

        Iinspect report

  - **make clean**

        Clean build                        

**First run without optimization**: 

  Gprof

```c
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
 97.63      5.66     5.66                             jacobi
  2.25      5.79     0.13                             initialize
  0.35      5.81     0.02                             error_check

```
  Perf

```c

    $ perf record -e cycles,instructions,cache-misses,cache-references,branch-misses,branch-instructions,mem-loads,mem-stores -g -- ./jacobi

Total Number of Iterations=101
Residual=6.243228E-11
Solution Error=1.332995E-04
[ perf record: Woken up 76 times to write data ]
[ perf record: Captured and wrote 16.736 MB perf.data (182615 samples) ]

    $perf report


Samples: 27K of event 'cycles:u', Event count (approx.): 20227583458
  Children      Self  Command  Shared Object     Symbol
+   99.91%     0.00%  jacobi   [unknown]         [.] 0x5541d68949564100
+   99.91%     0.00%  jacobi   libc-2.31.so      [.] __libc_start_main
+   99.91%     0.00%  jacobi   jacobi            [.] main
+   98.55%    98.11%  jacobi   jacobi            [.] jacobi
+    1.01%     0.96%  jacobi   jacobi            [.] initialize
     0.47%     0.47%  jacobi   [unknown]         [k] 0xffffffffac6010e0
     0.42%     0.42%  jacobi   jacobi            [.] error_check
     0.04%     0.04%  jacobi   [unknown]         [k] 0xffffffffac601c60
...
```

  Vtune

```c
...
Vectorization: 0.0% of Packed FP Operations
 | This code has floating point operations and is not vectorized. Consider
 | either recompiling the code with optimization options that allow
 | vectorization or using Intel Advisor to vectorize the loops.
 |
    Instruction Mix
        SP FLOPs: 0.0% of uOps
            Packed: 0.0% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 0.0% from SP FP
            Scalar: 0.0% from SP FP
        DP FLOPs: 47.9% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 100.0% from DP FP
             | This code has floating point operations and is not vectorized.
             | Consider either recompiling the code with optimization options
             | that allow vectorization or using Intel Advisor to vectorize the
             | loops.
             |
        x87 FLOPs: 0.0% of uOps
        Non-FP: 52.1% of uOps
    FP Arith/Mem Rd Instr. Ratio: 1.871
    FP Arith/Mem Wr Instr. Ratio: 6.544
...
```


**Second run with OpenMP+SIMD - No Auto-Vectorization**: 

  Gprof

```c
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
 98.56      6.83     6.83                             __gmon_start__
  1.30      6.92     0.09                             initialize
  0.29      6.94     0.02                             error_check

```
  Perf

```c

    $ perf record -e cycles,instructions,cache-misses,cache-references,branch-misses,branch-instructions,mem-loads,mem-stores -g -- ./jacobi

Total Number of Iterations=101
Residual=6.243228E-11
Solution Error=1.332995E-04
[ perf record: Woken up 68 times to write data ]
[ perf record: Captured and wrote 16.732 MB perf.data (197165 samples) ]

    $perf report

Samples: 30K of event 'cycles:u', Event count (approx.): 19759324578
  Children      Self  Command  Shared Object     Symbol
+   82.75%     0.00%  jacobi   [unknown]         [.] 0000000000000000
+   72.39%     0.00%  jacobi   libgomp.so.1.0.0  [.] 0x0000148f4f1bc986
+   54.81%    54.76%  jacobi   jacobi            [.] jacobi._omp_fn.1
+   41.04%    41.02%  jacobi   jacobi            [.] jacobi._omp_fn.0
+   23.47%     0.00%  jacobi   libgomp.so.1.0.0  [.] GOMP_parallel
+   13.33%     0.00%  jacobi   [unknown]         [.] 0x000055735a9950a0
+   10.14%     0.00%  jacobi   [unknown]         [.] 0x000055734b5710a0
+    1.91%     1.91%  jacobi   libgomp.so.1.0.0  [.] 0x000000000001e43a
+    1.91%     0.00%  jacobi   libgomp.so.1.0.0  [.] 0x0000148f4f1bf43a
+    1.37%     0.00%  jacobi   [unknown]         [.] 0x5541d68949564100
+    1.37%     0.00%  jacobi   libc-2.31.so      [.] __libc_start_main
+    1.37%     0.00%  jacobi   jacobi            [.] main
+    0.96%     0.96%  jacobi   jacobi            [.] initialize
     0.41%     0.41%  jacobi   jacobi            [.] error_check
     0.36%     0.36%  jacobi   libgomp.so.1.0.0  [.] 0x000000000001e43e
...
```

VTune

```c
vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./jacobi

...
Vectorization: 95.9% of Packed FP Operations
 | Using the latest vector instruction set can improve parallelism for this
 | code. Consider either recompiling the code with the latest instruction set or
 | using Intel Advisor to get vectorization help.
 |
    Instruction Mix
        SP FLOPs: 0.0% of uOps
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
        DP FLOPs: 27.4% of uOps
            Packed: 95.9% from DP FP
                128-bit: 95.9% from DP FP
                 | Using the latest vector instruction set can improve
                 | parallelism for this code. Consider either recompiling the
                 | code with the latest instruction set or using Intel Advisor
                 | to get vectorization help.
                 |
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 4.1% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 72.6% of uOps
    FP Arith/Mem Rd Instr. Ratio: 1.433
    FP Arith/Mem Wr Instr. Ratio: 4.388
...
```

**Second run with OpenMP+SIMD + Auto Vectorization**: 

```c
make MODE=vectorized
Compiling jacobi.c as jacobi with mode: vectorized
gcc               -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec -o jacobi jacobi.c -lm
jacobi.c:139:14: optimized: loop vectorized using 64 byte vectors
jacobi.c:143:39: optimized: loop vectorized using 64 byte vectors
jacobi.c:83:9: optimized: loop vectorized using 64 byte vectors
jacobi.c:83:9: optimized:  loop versioned for vectorization because of possible aliasing
jacobi.c:83:9: optimized: loop vectorized using 32 byte vectors
jacobi.c:131:17: optimized: basic block part vectorized using 64 byte vectors
jacobi.c:137:10: optimized: basic block part vectorized using 64 byte vectors

vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./jacobi

...
Vectorization: 97.3% of Packed FP Operations
    Instruction Mix
        SP FLOPs: 0.0% of uOps
            Packed: 0.0% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 0.0% from SP FP
            Scalar: 0.0% from SP FP
        DP FLOPs: 10.5% of uOps
            Packed: 97.3% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 97.3% from DP FP
            Scalar: 2.7% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 89.5% of uOps
    FP Arith/Mem Rd Instr. Ratio: 0.870
    FP Arith/Mem Wr Instr. Ratio: 1.456
...



```

### Statistics & Analysis


This benchmark compares the average real execution time of the Jacobi solver under different optimization setups and OpenMP thread counts. Speedup is computed relative to the non-optimized baseline.

| Setup                                     | Threads | Avg Real Time (s) | Speedup vs Baseline |
|------------------------------------------|---------|-------------------|---------------------|
| No optimization                          |   1     | 6.205             | 1.00× (baseline)    |
| OpenMP + SIMD                            |   4     | 1.918             | 3.23×               |
| OpenMP + SIMD + Auto Vectorization       |   4     | 1.752             | 3.54×               |
| OpenMP + SIMD + Auto Vectorization       |   8     | 1.867             | 3.32×               |


---


VTune analysis on main optimization attributes

| Setup                          | Packed DP FLOPs | Vector ISA Used | DP Scalar | FP Arith/Mem Rd Ratio | SIMD Width |
|--------------------------------|------------------|------------------|-----------|------------------------|------------|
| OpenMP + SIMD                  | 95.9%            | 128-bit          | 4.1%      | 1.433                  | Narrow     |
| OpenMP + SIMD + Auto Vec       | 97.3%            | 512-bit          | 2.7%      | 0.870                  | Wide       |

---

Profiling tool overview

 Tool     | Use Case                                | Outcome Summary                                      |
|----------|------------------------------------------|------------------------------------------------------|
| `gprof`  | Function-level CPU time breakdown        | Identified `jacobi()` as >97% bottleneck            |
| `perf`   | Low-level performance counters           | Confirmed hotspot and cycles in `jacobi()` loop     |
| `vtune`  | Vectorization, ILP, FP instruction mix   | Diagnosed scalar vs packed ops; guided SIMD tuning  |


### Conclusions

Parallelism (OpenMP) and vectorization (SIMD + AVX) combined offer a **3.5× performance improvement** for the Jacobi solver function. 


- **Vectorization Gain**: Adding compiler auto-vectorization improved vector width from **128-bit to 512-bit** SIMD lanes.
- **Instruction Mix Shift**:
  - DP FLOP scalar operations decreased → more loop vectorization.
  - **512-bit SIMD** operations dominate in the fully optimized version.
- **FP Arithmetic/Memory Ratio** dropped slightly with AVX-512, indicating better bandwidth usage and potentially more instruction mix.
- **Speedup from 4-thread parallelism + AVX-512** reaches **3.5×** over scalar baseline.
- **8-thread execution** does not scale much further due to **saturation or memory bandwidth** limits.


For future work, it would wworth it further investigation on:
- **Memory access tuning** (cache blocking).
- **NUMA-aware execution** for >8 threads.



## Labs1, Vectorization with OpenMP; 5. Vectorize & parallelize the code swim.c



- compute -c 4
- module load intel vtune
- export OMP_NUM_THREADS={1,4,8}

### Compiler and flags used:


- **Makefile**: Make/Gprof
  - **make MODE=profile**

        Basic mode

  - **make MODE=vectorized**

        To add auto-vectorizing flag to compiler

  - **make run-profile**

        Runs program to generate gmon.out

  - **make gprof-report**

        Analyzes and dumps result

  - **cat gprof-report.txt**

        Iinspect report

  - **make clean**

        Clean build                        

**First run without optimization**: 

  Gprof

```c
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
 40.28      0.33     0.33                             calc1
 31.73      0.59     0.26                             calc3
 28.07      0.82     0.23                             calc2

```
  Perf

```c

    $ perf record -e cycles,instructions,cache-misses,cache-references,branch-misses,branch-instructions,mem-loads,mem-stores -g -- ./swim

NUMBER OF POINTS IN THE X DIRECTION 512
NUMBER OF POINTS IN THE y DIRECTION 512
GRID SPACING IN THE X DIRECTION 25000.000000
GRID SPACING IN THE Y DIRECTION 25000.000000
TIME STEP 20.000000
TIME FILTER PARAMETER 0.001000
NUMBER OF ITERATIONS 200

Pcheck = 1.313569E+10
Ucheck = 5.215097E+04
Vcheck = 5.215168E+04

Pcheck = 1.313569E+10
Ucheck = 5.211337E+04
Vcheck = 5.215157E+04

Pcheck = 1.313569E+10
Ucheck = 5.209151E+04
Vcheck = 5.215175E+04

Pcheck = 1.313568E+10
Ucheck = 5.207779E+04
Vcheck = 5.215205E+04
[ perf record: Woken up 14 times to write data ]
[ perf record: Captured and wrote 2.250 MB perf.data (24531 samples) ]

    $perf report


Samples: 3K of event 'cycles:u', Event count (approx.): 2366140682
  Children      Self  Command  Shared Object     Symbol
+   99.36%     0.00%  swim     [unknown]         [.] 0x5541d68949564100
+   99.36%     0.00%  swim     libc-2.31.so      [.] __libc_start_main
+   99.36%     0.19%  swim     swim              [.] main
+   37.55%    37.55%  swim     swim              [.] calc1
+   31.23%    31.23%  swim     swim              [.] calc3
+   29.95%    29.93%  swim     swim              [.] calc2
     0.27%     0.27%  swim     swim              [.] initial
     0.06%     0.06%  swim     libm-2.31.so      [.] 0x000000000007d4be
     0.06%     0.00%  swim     libm-2.31.so      [.] 0x000014f8b5ba34be
...
```
  Vtune

```c
vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./swim

...
Vectorization: 0.0% of Packed FP Operations
 | This code has floating point operations and is not vectorized. Consider
 | either recompiling the code with optimization options that allow
 | vectorization or using Intel Advisor to vectorize the loops.
 |
    Instruction Mix
        SP FLOPs: 24.5% of uOps
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
        DP FLOPs: 12.5% of uOps
            Packed: 0.0% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 100.0% from DP FP
             | This code has floating point operations and is not vectorized.
             | Consider either recompiling the code with optimization options
             | that allow vectorization or using Intel Advisor to vectorize the
             | loops.
             |
        x87 FLOPs: 0.0% of uOps
        Non-FP: 63.0% of uOps
    FP Arith/Mem Rd Instr. Ratio: 2.407
    FP Arith/Mem Wr Instr. Ratio: 4.805
...
```

**Second run with OpenMP+SIMD - No Auto-Vectorization**:

  Gprof

```c
...
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
100.06      0.71     0.71                             __gmon_start__

...
```

```
  Perf

```c
...
Samples: 3K of event 'cycles:u', Event count (approx.): 2265911367
  Children      Self  Command  Shared Object     Symbol
+   72.59%     0.00%  swim     [unknown]         [.] 0000000000000000
+   67.44%     0.00%  swim     libgomp.so.1.0.0  [.] 0x000014be06014986
+   39.14%    39.14%  swim     swim              [.] calc1._omp_fn.0
+   31.33%    31.33%  swim     swim              [.] calc2._omp_fn.0
+   22.57%     0.03%  swim     libgomp.so.1.0.0  [.] GOMP_parallel
+   19.43%    19.43%  swim     swim              [.] calc3._omp_fn.0
+    9.80%     0.00%  swim     [unknown]         [.] 0x4c6c520000000000
+    7.79%     0.00%  swim     [unknown]         [.] 0x000000003ad1b717
+    3.87%     3.87%  swim     libgomp.so.1.0.0  [.] 0x000000000001e282
+    3.87%     0.00%  swim     libgomp.so.1.0.0  [.] 0x000014be06017282
+    1.79%     0.00%  swim     [unknown]         [.] 0x5541d68949564100
+    1.79%     0.17%  swim     swim              [.] main
+    1.79%     0.00%  swim     libc-2.31.so      [.] __libc_start_main
+    1.77%     1.77%  swim     libgomp.so.1.0.0  [.] 0x000000000001e43a
...

```

  Vtune

```c
vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./swim

...
Vectorization: 13.5% of Packed FP Operations
 | A significant fraction of floating point arithmetic instructions are scalar.
 | This indicates that the code was not fully vectorized. Use Intel Advisor to
 | see possible reasons why the code was not vectorized.
 |
    Instruction Mix
        SP FLOPs: 29.3% of uOps
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
        DP FLOPs: 10.2% of uOps
            Packed: 52.4% from DP FP
                128-bit: 52.4% from DP FP
                 | Using the latest vector instruction set can improve
                 | parallelism for this code. Consider either recompiling the
                 | code with the latest instruction set or using Intel Advisor
                 | to get vectorization help.
                 |
                256-bit: 0.0% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 47.6% from DP FP
             | A significant fraction of floating point arithmetic instructions
             | are scalar. This indicates that the code was not fully
             | vectorized. Use Intel Advisor to see possible reasons why the
             | code was not vectorized.
             |
        x87 FLOPs: 0.0% of uOps
        Non-FP: 60.5% of uOps
    FP Arith/Mem Rd Instr. Ratio: 2.446
    FP Arith/Mem Wr Instr. Ratio: 6.282
...

```

**Second run with OpenMP+SIMD + Auto Vectorization**: 

```c
make MODE=vectorized
Compiling swim.c as swim with mode: vectorized
gcc               -O3 -fopenmp -march=native -ftree-vectorize -fopt-info-vec -o swim swim.c -lm
swim.c: In function ‘initial’:
swim.c:237:36: optimized: loop vectorized using 64 byte vectors
swim.c:237:36: optimized: loop vectorized using 32 byte vectors
swim.c:276:37: optimized: loop vectorized using 64 byte vectors
swim.c:276:37: optimized: loop vectorized using 32 byte vectors
swim.c:324:30: optimized: loop vectorized using 64 byte vectors
swim.c:324:30: optimized: loop vectorized using 32 byte vectors
swim.c:206:5: optimized: loop vectorized using 64 byte vectors
swim.c:206:5: optimized:  loop versioned for vectorization because of possible aliasing
swim.c:206:5: optimized: loop vectorized using 32 byte vectors
swim.c:200:9: optimized: loop vectorized using 64 byte vectors
swim.c:200:9: optimized: loop vectorized using 32 byte vectors
swim.c:245:5: optimized: loop vectorized using 64 byte vectors
swim.c:245:5: optimized:  loop versioned for vectorization because of possible aliasing
swim.c:245:5: optimized: loop vectorized using 32 byte vectors
swim.c:283:5: optimized: loop vectorized using 64 byte vectors
swim.c:283:5: optimized:  loop versioned for vectorization because of possible aliasing
swim.c:283:5: optimized: loop vectorized using 32 byte vectors
swim.c:333:5: optimized: loop vectorized using 64 byte vectors
swim.c:333:5: optimized:  loop versioned for vectorization because of possible aliasing
swim.c:333:5: optimized: loop vectorized using 32 byte vectors
swim.c:122:9: optimized: loop vectorized using 64 byte vectors
swim.c:122:9: optimized: loop vectorized using 32 byte vectors


vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./swim

...
Vectorization: 79.8% of Packed FP Operations
    Instruction Mix
        SP FLOPs: 15.3% of uOps
            Packed: 94.4% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 94.4% from SP FP
            Scalar: 5.6% from SP FP
        DP FLOPs: 20.3% of uOps
            Packed: 68.9% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 0.0% from DP FP
                512-bit: 68.9% from DP FP
            Scalar: 31.1% from DP FP
             | A significant fraction of floating point arithmetic instructions
             | are scalar. This indicates that the code was not fully
             | vectorized. Use Intel Advisor to see possible reasons why the
             | code was not vectorized.
             |
        x87 FLOPs: 0.0% of uOps
        Non-FP: 64.5% of uOps
    FP Arith/Mem Rd Instr. Ratio: 1.496
    FP Arith/Mem Wr Instr. Ratio: 6.847
...

```


### Statistics & Analysis


This table summarizes the average real time over 3 runs per optimization setup and the corresponding speedup compared to the baseline (no optimization).

| Setup                                      | Threads | Avg Real Time (s) | Speedup vs Baseline |
|-------------------------------------------|---------|-------------------|---------------------|
| No optimization (baseline)                |    1    | 0.857             | 1.00×               |
| OpenMP + SIMD                             |    4    | 0.215             | 3.99×               |
| OpenMP + SIMD                             |    8    | 0.329             | 2.61×               |
| OpenMP + SIMD + auto vectorization        |    4    | 0.114             | 7.52×               |
| OpenMP + SIMD + auto vectorization        |    8    | 0.152             | 5.64×               |

----



VTune analysis on main optimization attributes

| Setup                            | Packed FP (%) | Notable Observation |
|----------------------------------|---------------|----------------------|
| No Optimization                  | 0%            | Fully scalar code. |
| OpenMP + SIMD                    | 13.5%         | Limited due to aliasing and scalar loops. |
| OpenMP + SIMD + Auto-Vectorize  | **79.8%**     | Significant vectorization of both SP and DP ops using **512-bit SIMD**. |

----



Profiling tools overview

Baseline Observations (No Optimization)

| Tool   | Key Finding |
|--------|-------------|
| `gprof` | `calc1`, `calc2`, and `calc3` account for **100% of execution time**. |
| `perf` | Confirms top CPU cycles consumed by `calc1`, `calc2`, `calc3`. |
| `vtune` | **No vectorization detected**, 100% scalar FP operations. |




### Conclusions


- Best speedup achieved with OpenMP + SIMD + auto vectorization using 4 threads.

- 8 threads had slightly worse performance, likely due to:

    Thread contention or oversubscription

    NUMA/memory bandwidth saturation

- Auto vectorization combined with hand-annotated OpenMP provided the highest overall performance. Reducing runtime by ~86%.
- Manual SIMD alone is insufficient — aliasing and loop dependencies can block vectorization.

- Auto-vectorization with aggressive compiler flags (-O3 -march=native -fopt-info-vec) is essential to unlock full hardware potential.

- OpenMP parallelism + loop vectorization is synergistic — vector units process data faster per thread, OpenMP scales across cores.

- Profiling first is crucial — gprof, perf, and VTune consistently pointed to same hotspots and confirmed gains.




## Labs1, Thread Aﬃnity; 1: heat.c


Performance & Speedup Analysis: Matrix Multiplication (D = A × Bᵗ)

- compute -c 64 --mem 246G
- module load intel vtune
- export OMP_NUM_THREADS={2,4,8}
- export OMP_PLACES="{0:16},{16:16},{32:16},{48:16}"
- export OMP_PLACES={threads,cores,sockets}
- export OMP_PROC_BIND={master, close, spread}



### Compiler and flags used:


- **Makefile**: Make/Gprof
  - **make MODE=profile**

        Basic mode

  - **make MODE=vectorized**

        To add auto-vectorizing flag to compiler

  - **make run-profile**

        Runs program to generate gmon.out

  - **make gprof-report**

        Analyzes and dumps result

  - **cat gprof-report.txt**

        Iinspect report

  - **make clean**

        Clean build                        

**Run with OpenMP + SIMD**: 


```c
 make
Compiling heat.c as heat with mode: basic
gcc               -O2 -fopenmp -fopt-info-vec -o heat heat.c -lm
heat.c:59:18: optimized: loop vectorized using 16 byte vectors
heat.c:56:20: optimized: loop vectorized using 16 byte vectors
heat.c:44:3: optimized: loop vectorized using 16 byte vectors

```



### Statistics & Analysis

Setup: baseline, no optimization
time ./heat 5000000
Tiempo  7.455 s.
Result = 1

real	0m7.470s
user	0m7.440s
sys	0m0.004s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  7.491 s.
Result = 1

real	0m7.504s
user	0m7.476s
sys	0m0.004s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  7.505 s.
Result = 1

real	0m7.521s
user	0m7.488s
sys	0m0.007s

Setup: OpenMP + SIMD, OMP_NUM_THREADS=2, OMP_PROC_BIND=master
time ./heat 5000000
Tiempo  2.438 s.
Result = 1

real	0m2.446s
user	0m4.861s
sys	0m0.008s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  2.431 s.
Result = 1

real	0m2.440s
user	0m4.848s
sys	0m0.009s
[curso370@c206-8 heat]$ time ./heat 5000000

Tiempo  2.466 s.
Result = 1

real	0m2.472s
user	0m4.916s
sys	0m0.007s


Setup: OpenMP + SIMD, OMP_NUM_THREADS=4, OMP_PROC_BIND=master
time ./heat 5000000
Tiempo  1.619 s.
Result = 1

real	0m1.631s
user	0m6.445s
sys	0m0.007s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  1.631 s.
Result = 1

real	0m1.636s
user	0m6.491s
sys	0m0.010s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  1.573 s.
Result = 1

real	0m1.579s
user	0m6.257s
sys	0m0.013s

Setup: OpenMP + SIMD, OMP_NUM_THREADS=8, OMP_PROC_BIND=master
time ./heat 5000000
Tiempo  1.090 s.
Result = 1

real	0m1.095s
user	0m8.664s
sys	0m0.012s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  1.104 s.
Result = 1

real	0m1.110s
user	0m8.777s
sys	0m0.010s
[curso370@c206-8 heat]$ time ./heat 5000000
Tiempo  1.059 s.
Result = 1

real	0m1.064s
user	0m8.417s
sys	0m0.012s



## Overall Future Work

### HPC Tools - Compilation, profiling and optimization of HPC Software
  - "3. The compiler: a key tool to exploit HPC resources"
  - "10 - ProfOpt03.pdf"
  - cc -S -fverbose-asm -O3 -march=native -g saxpy.c -o saxpy.s
  - as -alhnd saxpy.s > saxpy.lst
  - cat saxpy.lst


## References 
[Santiago de Compostela - HPC - HPC Tools - Profiling tools for SpMV](https://github.com/TIAGOOOLIVEIRA/Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp/tree/main/hpc_tools/spmv)
