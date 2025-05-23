# OpenMP and MPI Labs Assignment

## Labs1, 1. The code multf.c performs the product of matrices

Performance & Speedup Analysis: Matrix Multiplication (D = A × Bᵗ)

- compute -c 4
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


## Labs1, 2,3. Parallelize and vectorize saxpy.c  

- compute -c 4
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

## Future Work

HPC Tools - Compilation, profiling and optimization of HPC Software
3. The compiler: a key tool to exploit HPC resources
"10 - ProfOpt03.pdf"
cc -S -fverbose-asm -O3 -march=native -g saxpy.c -o saxpy.s
as -alhnd saxpy.s > saxpy.lst
cat saxpy.lst


## References 
[Santiago de Compostela - HPC - HPC Tools - Profiling tools for SpMV](https://github.com/TIAGOOOLIVEIRA/Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp/tree/main/hpc_tools/spmv)
