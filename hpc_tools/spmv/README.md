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
| Ref         | 22.3   | 22       | 22     | 22.7      |

**Execution time scale is in milliseconds for all benchmarking

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

$module load intel imkl


    O0
    icc -O0 spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

    O0 --> MKL
    icc -O0 spmv_mkl.c my_sparseCSR_mkl.c timer.c -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -o spmv_mkl

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 760      | 130    | 98     | 109    |  23   |
| #2    | 766      | 144    | 100    | 112    |  22   |
| #3    | 772      | 133    | 101    | 111    |  23   |
| **avg** | 766      | 135.7  | 99.7  | 110.7  | 22.3 |


    O2 -fno-tree-vectorize
    icc -O2 -fno-tree-vectorize spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

    O2 -fno-tree-vectorize --> MKL
    icc -O2 -fno-tree-vectorize spmv_mkl.c my_sparseCSR_mkl.c timer.c -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -o spmv_mkl

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 338      | 72     | 44     | 28     | 22    |
| #2    | 337      | 72     | 33     | 28     | 22    |
| #3    | 338      | 72     | 34     | 28     | 22    |
| **avg** | 337.7    | 72     | 37     | 28     | 22    |


    O3
    icc -O3 spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

    O3 --> MKL
    icc -O3 spmv_mkl.c my_sparseCSR_mkl.c timer.c -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -o spmv_mkl

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 151      | 72     | 24     | 28     | 22    |
| #2    | 149      | 72     | 22     | 27     | 22    |
| #3    | 150      | 72     | 22     | 28     |  22   |
| **avg** | 150      | 72     | 22.7   | 27.7   | 22    |


    Ofast
    icc -Ofast spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -lgsl -lgslcblas -lpthread -lm -ldl -o spmv

    Ofast --> MKL
    icc -Ofast spmv_mkl.c my_sparseCSR_mkl.c timer.c -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -o spmv_mkl

|       | My_Dense | My_coo | My_csr | My_csc | Ref   |
|-------|----------|--------|--------|--------|-------|
| #1    | 150      | 72     | 22     | 28     | 22    |
| #2    | 148      | 72     | 22     | 29     | 23    |
| #3    | 151      | 71     | 22     | 28     | 23    |
| **avg** | 149.7    | 71.7   | 22     | 28.3   | 22.7  |


________________________________________________________________________________________________________
- Make

  -- make -f Makefile.gcc
  - make -f Makefile.gcc spmv_O2
  - make -f Makefile.gcc spmv_O0
  - make -f Makefile.gcc spmv_O3
  - make -f Makefile.gcc spmv_Ofast
  - make -f Makefile.gcc clean


  -- make -f Makefile.icc
  - make -f Makefile.icc spmv_O0
  - make -f Makefile.icc spmv_O2
  - make -f Makefile.icc spmv_O3
  - make -f Makefile.icc spmv_Ofast
  - make -f Makefile.icc clean


  -- make -f Makefile.icc.mkl
  - make -f Makefile.icc.mkl spmv_mkl_O0
  - make -f Makefile.icc.mkl spmv_mkl_O2
  - make -f Makefile.icc.mkl spmv_mkl_O3
  - make -f Makefile.icc.mkl spmv_mkl_Ofast
  - make -f Makefile.icc.mkl clean



________________________________________________________________________________________________________
## Profiling tools and tecniques for optimization

- Profiling (to spot optimization made on the code by the compiler):
  - Likwid
  - Valgrind
  - VTune
  - Perf

$ module load intel vtune imkl valgrind


- Compile for the best possible vectorization and profiling enabled
    
    $ gcc -pg -O3 -lopenblas -lgsl -lgslcblas spmv.c timer.c my_dense.c my_sparse.c my_csr.c my_coo.c my_csc.c -o spmv
    

    $ icc -pg -O3 spmv_mkl.c my_sparseCSR_mkl.c timer.c -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -o spmv_mkl


### Optimization
- VTune

    $ vtune -collect performance-snapshot -collect memory-access -collect hotspots -collect threading -- ./spmv


```c
vtune: Collection started. To stop the collection, either press CTRL-C or enter from another console window: vtune -r /mnt/netapp2/Home_FT2/home/ulc/cursos/curso370/hpctools/2025/git_emilio/spmv/r000ps -command stop.
Matriz size: 16384 x 16384 (268435456 elements)
26837519 non-zero elements (10.00%)

Dense computation
----------------
Time taken by CBLAS dense computation: 957 ms
Time taken by my dense matrix-vector product: 371 ms
Result is ok!

Sparse computation
------------------
Time taken by convert_to_gsl (Ref table 1:gsl-sparse): 7599 ms
Time taken by compute_sparse (Ref table 1:gsl-matmul): 74 ms
Result is correct for my_sparse!

CSR Sparse computation
------------------
Time taken by conversion to CSR computation: 960 ms
Time taken by CSR sparse computation: 34 ms
Result is ok for CSR sparse!

COO Sparse computation
------------------
Time taken by conversion to COO: 999 ms
Time taken by COO sparse computation: 85 ms
Result is ok for COO sparse!

CSC Sparse computation
------------------
Time taken by conversion to CSC: 4549 ms
Time taken by CSC sparse computation: 32 ms
Result is ok for CSC sparse!
vtune: Collection stopped.
vtune: Using result path `/mnt/netapp2/Home_FT2/home/ulc/cursos/curso370/hpctools/2025/git_emilio/spmv/r000ps'
vtune: Executing actions 75 % Generating a report                              Elapsed Time: 19.836s
    IPC: 1.193
    DP GFLOPS: 0.085
    Average CPU Frequency: 3.380 GHz
Logical Core Utilization: 0.7% (0.868 out of 128)
 | The metric value is low, which may signal a poor logical CPU cores
 | utilization. Consider improving physical core utilization as the first step
 | and then look at opportunities to utilize logical cores, which in some cases
 | can improve processor throughput and overall performance of multi-threaded
 | applications.
 |
    Physical Core Utilization: 1.3% (0.861 out of 64)
     | The metric value is low, which may signal a poor physical CPU cores
     | utilization caused by:
     |     - load imbalance
     |     - threading runtime overhead
     |     - contended synchronization
     |     - thread/process underutilization
     |     - incorrect affinity that utilizes logical cores instead of physical
     |       cores
     | Run the HPC Performance Characterization analysis to estimate the
     | efficiency of MPI and OpenMP parallelism or run the Locks and Waits
     | analysis to identify parallel bottlenecks for other parallel runtimes.
     |
Microarchitecture Usage: 24.6% of Pipeline Slots
 | You code efficiency on this platform is too low.
 | 
 | Possible cause: memory stalls, instruction starvation, branch misprediction
 | or long latency instructions.
 | 
 | Next steps: Run Microarchitecture Exploration analysis to identify the cause
 | of the low microarchitecture usage efficiency.
 |
    Retiring: 24.6% of Pipeline Slots
    Front-End Bound: 9.8% of Pipeline Slots
    Bad Speculation: 15.8% of Pipeline Slots
     | A significant proportion of pipeline slots containing useful work are
     | being cancelled. This can be caused by mispredicting branches or by
     | machine clears. Note that this metric value may be highlighted due to
     | Branch Resteers issue.
     |
    Back-End Bound: 49.8% of Pipeline Slots
     | A significant portion of pipeline slots are remaining empty. When
     | operations take too long in the back-end, they introduce bubbles in the
     | pipeline that ultimately cause fewer pipeline slots containing useful
     | work to be retired per cycle than the machine is capable to support. This
     | opportunity cost results in slower execution. Long-latency operations
     | like divides and memory operations can cause this, as can too many
     | operations being directed to a single execution port (for example, more
     | multiply operations arriving in the back-end per cycle than the execution
     | unit can support).
     |
        Memory Bound: 29.6% of Pipeline Slots
         | The metric value is high. This can indicate that the significant
         | fraction of execution pipeline slots could be stalled due to demand
         | memory load and stores. Use Memory Access analysis to have the metric
         | breakdown by memory hierarchy, memory bandwidth information,
         | correlation by memory objects.
         |
            L1 Bound: 5.5% of Clockticks
             | This metric shows how often machine was stalled without missing
             | the L1 data cache. The L1 cache typically has the shortest
             | latency. However, in certain cases like loads blocked on older
             | stores, a load might suffer a high latency even though it is
             | being satisfied by the L1. Note that this metric value may be
             | highlighted due to DTLB Overhead or Cycles of 1 Port Utilized
             | issues.
             |
            L2 Bound: 7.5% of Clockticks
             | This metric shows how often machine was stalled on L2 cache.
             | Avoiding cache misses (L1 misses/L2 hits) will improve the
             | latency and increase performance.
             |
            L3 Bound: 0.7% of Clockticks
                L3 Latency: 5.3% of Clockticks
            DRAM Bound: 21.1% of Clockticks
             | This metric shows how often CPU was stalled on the main memory
             | (DRAM). Caching typically improves the latency and increases
             | performance.
             |
                Memory Bandwidth: 31.2% of Clockticks
                 | Issue: A significant fraction of cycles was stalled due to
                 | approaching bandwidth limits of the main memory (DRAM).
                 | 
                 | Tips: Improve data accesses to reduce cacheline transfers
                 | from/to memory using these possible techniques:
                 |     - Consume all bytes of each cacheline before it is
                 |       evicted (for example, reorder structure elements and
                 |       split non-hot ones).
                 |     - Merge compute-limited and bandwidth-limited loops.
                 |     - Use NUMA optimizations on a multi-socket system.
                 | 
                 | Note: software prefetches do not help a bandwidth-limited
                 | application.
                 |
                Memory Latency: 8.1% of Clockticks
                    Local DRAM: 40.5% of Clockticks
                    Remote DRAM: 3.0% of Clockticks
                    Remote Cache: 0.6% of Clockticks
            Store Bound: 0.1% of Clockticks
        Core Bound: 20.3% of Pipeline Slots
         | This metric represents how much Core non-memory issues were of a
         | bottleneck. Shortage in hardware compute resources, or dependencies
         | software's instructions are both categorized under Core Bound. Hence
         | it may indicate the machine ran out of an OOO resources, certain
         | execution units are overloaded or dependencies in program's data- or
         | instruction- flow are limiting the performance (e.g. FP-chained long-
         | latency arithmetic operations).
         |
Memory Bound: 29.6% of Pipeline Slots
 | The metric value is high. This can indicate that the significant fraction of
 | execution pipeline slots could be stalled due to demand memory load and
 | stores. Use Memory Access analysis to have the metric breakdown by memory
 | hierarchy, memory bandwidth information, correlation by memory objects.
 |
    Cache Bound: 13.7% of Clockticks
    DRAM Bound: 21.1% of Clockticks
     | The metric value is high. This indicates that a significant fraction of
     | cycles could be stalled on the main memory (DRAM) because of demand loads
     | or stores.
     |
     | The code is memory bandwidth bound, which means that there are a
     | significant fraction of cycles during which the bandwidth limits of the
     | main memory are being reached and the code could stall. Review the
     | Bandwidth Utilization Histogram to estimate the scale of the issue.
     | Improve data accesses to reduce cacheline transfers from/to memory using
     | these possible techniques: 1) consume all bytes of each cacheline before
     | it is evicted (for example, reorder structure elements and split non-hot
     | ones); 2) merge compute-limited and bandwidth-limited loops; 3) use NUMA
     | optimizations on a multi-socket system.
     |
    NUMA: % of Remote Accesses: 5.1%
Vectorization: 11.8% of Packed FP Operations
    Instruction Mix
        SP FLOPs: 0.0% of uOps
            Packed: 0.0% from SP FP
                128-bit: 0.0% from SP FP
                256-bit: 0.0% from SP FP
                512-bit: 0.0% from SP FP
            Scalar: 0.0% from SP FP
        DP FLOPs: 1.7% of uOps
            Packed: 11.8% from DP FP
                128-bit: 0.0% from DP FP
                256-bit: 11.8% from DP FP
                512-bit: 0.0% from DP FP
            Scalar: 88.2% from DP FP
        x87 FLOPs: 0.0% of uOps
        Non-FP: 98.3% of uOps
    FP Arith/Mem Rd Instr. Ratio: 0.065
    FP Arith/Mem Wr Instr. Ratio: 0.177
Collection and Platform Info
    Application Command Line: ./spmv 
    Operating System: 4.18.0-305.3.1.el8_4.x86_64 \S Kernel \r on an \m 
    Computer Name: login210-19
    Result Size: 3.5 MB 
    Collection start time: 22:34:12 12/02/2025 UTC
    Collection stop time: 22:34:32 12/02/2025 UTC
    Collector Type: Driverless Perf per-process counting
    CPU
        Name: Intel(R) Xeon(R) Processor code named Icelake
        Frequency: 2.200 GHz
        Logical CPU Count: 128
        Cache Allocation Technology
            Level 2 capability: not detected
            Level 3 capability: available

Recommendations:
    Hotspots: Start with Hotspots analysis to understand the efficiency of your algorithm.
     | Use Hotspots analysis to identify the most time consuming functions.
     | Drill down to see the time spent on every line of code.
    Threading: There is poor utilization of logical CPU cores (0.7%) in your application.
     |  Use Threading to explore more opportunities to increase parallelism in
     | your application.
    Microarchitecture Exploration: There is low microarchitecture usage (24.6%) of available hardware resources.
     | Run Microarchitecture Exploration analysis to analyze CPU
     | microarchitecture bottlenecks that can affect application performance.
    Memory Access: The Memory Bound metric is high  (29.6%). A significant fraction of execution pipeline slots could be stalled due to demand memory load and stores.
     | Use Memory Access analysis to measure metrics that can identify memory
     | access issues.

If you want to skip descriptions of detected performance issues in the report,
enter: vtune -report summary -report-knob show-issues=false -r <my_result_dir>.
Alternatively, you may view the report in the csv format: vtune -report
<report_name> -format=csv.
vtune: Executing actions 100 % done   