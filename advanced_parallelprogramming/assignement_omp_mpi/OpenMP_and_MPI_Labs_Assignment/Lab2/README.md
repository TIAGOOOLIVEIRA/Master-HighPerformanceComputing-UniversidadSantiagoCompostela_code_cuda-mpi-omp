# Labs 2.- MPI: Nonblocking Collective Communications


## MPI: pi_integral.c

After implementing the MPI nonblocking collective operations for pi_integral.c the shell from Lab1 pi_integral.c was adapted so the same analysis on the execution could be done to perform the comparison between both approaches.

To compile
- #gcc -c ../../Lab1/shared/place_report_mpi.c -o place_report_mpi.o
- #mpicc -fopenmp pi_integral.c ../../Lab1/shared/place_report_mpi.c -o pi_integral -I ../../Lab1/shared -lpthread

For the interactive mode, the pi_integral.sh can be changed its access mode as follows
- #chmod +x pi_integral.sh
- #./pi_integral.sh
- #mpirun -np 4 ./pi_integral 1000000000

Otherwise it is just the matter to submit the file as a job to the slurm scheduler
- #sbatch pi_integral.sh

The speedup analysis for the pi_integral.c in the Lab1 is taken into account to perform the analysis comparison as follows.


### MPI+OpenMP Speedup Table – Nonblocking Collectives

- Nonblocking MPI

| MPI × OMP | Total Cores | Time (s) | Speedup vs Baseline | Observations                         |
|-----------|-------------|----------|----------------------|--------------------------------------|
| 1 × 1     | 1           | 6.200    | 1.00× (baseline)     | Serial, blocking version baseline |
| 2 × 8     | 16          | 0.056433 | 109.90×              | Superb hybrid scaling             |
| 4 × 4     | 16          | 0.049028 | 126.46×              | Best hybrid balance (nonblocking) |
| 8 × 2     | 16          | 0.050750 | 122.22×              | Strong hybrid, low sync overhead  |
| 16 × 1    | 16          | 0.048685 | 127.37×              | All-MPI shines with nonblocking   |


- Comparative Analysis: Blocking vs Nonblocking MPI

  __comparison based on the analysis in /Lab1/README.md "Labs1, Hybrid Programming; 1: pi_integral.c"__

| Feature                      | Blocking MPI Version         | Nonblocking MPI Version               |
| ---------------------------- | ---------------------------- | ------------------------------------- |
| **Best Time**                | 0.750 s (4×4)                | 0.048685 s (16×1)                     |
| **Best Speedup**             | 8.26×                        | 127.37×                               |
| **Synchronization Overhead** | Significant at 16×1 (5.12×)  | Drastically reduced in 16×1 (127×)    |
| **Hybrid Balance**           | 4×4 had the best performance | All layouts perform similarly well    |
| **MPI-only Scalability**     | Poor (due to blocking sync)  | Excellent (nonblocking hides latency) |


### Conclusions
- Nonblocking collectives (e.g., MPI_Ireduce + MPI_Wait) dramatically reduce communication stalls, particularly in pure MPI (no threads) scenarios.

- In the blocking version, 16×1 suffered from heavy synchronization and lack of intra-node threading.

- With nonblocking, 16×1 became the fastest configuration, indicating that latency was effectively hidden and collective progress overlapped with computation or other processes.

- Hybrid configurations (4×4, 8×2) still perform excellently — nonblocking collectives provide more consistently high efficiency across layouts.


## MPI: dotprod.c

To compile
- #gcc -c ../../Lab1/shared/place_report_mpi.c -o place_report_mpi.o
- #mpicc -fopenmp dotprod.c ../../Lab1/shared/place_report_mpi.c -o dotprod -I ../../Lab1/shared -lpthread

For the interactive mode, the pi_integral.sh can be changed its access mode as follows
- #chmod +x dotprod.sh
- #./dotprod.sh
- #mpirun -np 4 ./dotprod 1000000000

Otherwise it is just the matter to submit the file as a job to the slurm scheduler
- #sbatch dotprod.sh


The speedup analysis for the dotprod.c in the Lab1 is taken into account to perform the analysis comparison as follows.


### MPI+OpenMP Speedup Table – Nonblocking Collectives

- Nonblocking MPI

| MPI × OMP | Total Cores | Time (s)   | Speedup vs Baseline | Observations                            |
| --------- | ----------- | ---------- | ------------------- | --------------------------------------- |
| 1 × 1     | 1           | 6.254      | 1.00× (baseline)    | Serial baseline                      |
| 2 × 8     | 16          | 0.856      | 7.31×               | Very good threading per rank         |
| 4 × 4     | 16          | 0.784      | 7.97×               | Best hybrid balance                  |
| 8 × 2     | 16          | 0.933      | 6.70×               | MPI comm cost rising                 |
| 16 × 1    | 16          | ❌ SEGFAULT | —                   | Failed: memory access likely invalid |


- Comparative Analysis: Blocking vs Nonblocking MPI

  __comparison based on the analysis in /Lab1/README.md "Labs1, Hybrid Programming; 2: dotprod.c"__

| **Configuration (MPI×OMP)** | **Blocking Avg Time (s)** | **Blocking Speedup** | **Nonblocking Avg Time (s)** | **Nonblocking Speedup** |
| --------------------------- | ------------------------- | -------------------- | ---------------------------- | ----------------------- |
| 1 × 1 (baseline)            | 0.34300                   | 1.00×                | 6.25400                      | 1.00×                   |
| 2 × 8                       | 0.20877                   | 1.64×                | 0.85600                      | 7.31×                   |
| 4 × 4                       | 0.11792                   | 2.91×                | 0.78400                      | 7.97×                   |
| 8 × 2                       | 0.09237                   | 3.71×                | 0.93300                      | 6.70×                   |
| 16 × 1                      | 0.08577                   | 4.00×                | ❌ SEGFAULT                   | —                       |



### Conclusions
- Nonblocking collectives show much higher raw speedup (e.g., 7.97× at 4×4 vs 2.91× for blocking).

- Blocking collectives are slower, even though their baseline time (0.343s) is far below the nonblocking baseline (6.254s). Likely due to: Smaller problem size, Tighter node affinity, Less memory pressure

- Blocking version shows consistent improvement with increasing cores (1×1 → 16×1).

- Nonblocking version crashes at 16×1 due to poor memory bounds checking — chunking logic must be fixed.

- Later I realized that the fault in the (16 x 1) setup is due to the fact that I need to scale up the slurm job reserved capacity

  - this (#SBATCH --ntasks=16; #SBATCH --cpus-per-task=8; #SBATCH --mem=32G)
  - instead of this (#SBATCH --ntasks=16; #SBATCH --cpus-per-task=1; #SBATCH --mem=16G)

## MPI: mxnvm.c

A more simpllified benchmark analysis sticking to 3 execution samples per version of the matrix-vector multiplication program (mxvnm_*).
Important to mention that also the calculation per process leverage OpenMP for local parallelism.


To compile
- #mpicc -fopenmp mxvnm_collectiveblocking.c -o mxvnm_collectiveblocking
- #mpicc -fopenmp mxvnm_collectivenonblocking.c -o mxvnm_collectivenonblocking
- #gcc -fopenmp mxvnm_noncollective.c -o mxvnm_noncollective


To execute and collect statistics
- #time ./mxvnm_noncollective 5000 5000

- #export OMP_NUM_THREADS=4

- #mpirun -np 4 ./mxvnm_collectiveblocking 5000 5000
- #mpirun -np 4 ./mxvnm_collectivenonblocking 5000 5000



### MPI+OpenMP Speedup Table – Nonblocking Collectives

| Version               | Parallel Setup         | Avg Time (s) | Speedup vs Non-Collective (1×1) |
| --------------------- | ---------------------- | ------------ | ------------------------------- |
| `mxvnm_noncollective` | 1 MPI × 1 OMP (serial) | 0.1600       | 1.00× (baseline)                |
| `mxvnm_collectiveblocking`      | 4 MPI × 4 OMP          | 0.00477      | 33.54×                          |
| `mxvnm_collectivenonblocking`   | 4 MPI × 4 OMP          | 0.00461      | 34.72×                          |



### Conclusions

- Both blocking and non-blocking collectives offer dramatic speedup (~34×) over the single-threaded baseline.

- The non-blocking variant is slightly faster (~2-3%) than the blocking one — as expected due to early computation overlap.

- The consistency of output (y[0] and y[N-1]) across runs confirms numerical correctness.


## MPI: sqrt.c

Enabling Pipelining/Overlapping:

- The ability of MPI_I... calls to return immediately is crucial for overlapping. This allows the application to proceed with computation or other communication while the initiated collective operation progresses in the background
- For optimal performance and true overlapping, it is often necessary for the MPI library to have an asynchronous progress engine
- This can involve helper threads that continue to progress MPI operations while the main application threads compute. For example, with MPICH, setting MPICH_ASYNC_PROGRESS=1 can enable this, requiring MPI_THREAD_MULTIPLE support
- The concept of "weak local" progress means that an MPI operation might only complete when another MPI call is made that enables progress
- Therefore, it's often necessary to either frequently call MPI_Test() or use non-standard asynchronous progress mechanisms to ensure active overlapping.

  ***References***
    - Understanding MPI on Cray XC30; Basic information about Cray's MPI implementation (XC30_1-05-Cray_MPI.pdf)
    - Introduction to the Message Passing Interface (MPI); University of Stuttgart, High-Performance Computing-Center Stuttgart (HLRS) (mpi_3.1_rab_2023-JSC.pdf)


To compile
 - #mpicc -Wall -Wextra  -fopenmp sqrt.c -o sqrt -lm
 - #mpirun -np 4 ./sqrt 100000

### MPI+OpenMP Speedup Table – Nonblocking Collectives

To execute and benchmark

 - #mpirun -np 4 ./sqrt 100000 1000
- #mpirun -np 4 ./sqrt 100000 100
- #mpirun -np 4 ./sqrt 100000 10
- #mpirun -np 4 ./sqrt 100000 5


***This table compares the average execution time (in seconds) of the pipelined square root computation using MPI_Igather, varying the number of pipeline steps***

| Pipeline Steps | Avg Time (s) | Test Sum (per run) | Observations                                                   |
| -------------- | ------------ | ------------------ | -------------------------------------------------------------- |
| 1000           | 0.01136      | 300900             | High overhead from too many steps – poor pipelining efficiency |
| 100            | 0.00137      | 309000             | Dramatic speedup – better overlap balance                      |
| 10             | 0.00113      | 390000             | Near-optimal performance – minimal communication stalls        |
| 5              | 0.000935     | 480000             | Best timing – minimal pipeline setup cost                      |

### Conclusions

- Overhead with high steps: At 1000 steps, overhead dominates – too many small nonblocking operations reduce performance due to frequent buffer management and synchronization cost.

- Balanced pipelining: Around 100 or 10 steps, pipelining becomes efficient. MPI can overlap compute/comm, reducing wall time.

- Fewer steps (~5): Best raw performance but limits the granularity of pipelining. Some overlapping may be lost if chunks are too large to compute while communication occurs.

### Future work
***MPI - Async Progress Engine Support***


Asynchronous progress in MPI enables overlapping communication and computation, crucial for pipelined and nonblocking collective operations.

MPI libraries such as MPICH and Intel® MPI offer asynchronous progress engines, which rely on helper threads that continue progressing MPI operations independently of the main thread.

To activate this feature, environment variables must be configured:

    MPICH_ASYNC_PROGRESS=1 — enables the helper thread in MPICH.

    MPICH_NEMESIS_ASYNC_PROGRESS=1 — required on systems like Cray XC.

    MPICH_MAX_THREAD_SAFETY=multiple — ensures thread-safe MPI behavior.

Applications must initialize MPI using MPI_Init_thread(..., MPI_THREAD_MULTIPLE, ...) to ensure compatibility.

MPI engines generally rely on user MPI calls (e.g., MPI_Test, MPI_Wait) to make communication progress. Without an async engine, progress may stall if no calls are made.

Intel® MPI also supports this mechanism but requires tuning, and its use is not free, as helper threads introduce overhead and contention risks.

These async engines are essential for high-performance computing, especially when using nonblocking collectives (e.g., MPI_Igather, MPI_Iscatter) in pipelined patterns.

***Items to investigate***

The performance impact of enabling MPICH_ASYNC_PROGRESS=1 under various thread-count and message-size scenarios.

The overhead trade-offs introduced by helper threads and their interactions with OpenMP regions and CPU core binding.

Extend experiments to include other MPI distributions (e.g., Intel® MPI, MVAPICH2) and validate their respective async progress implementations.

Explore strategies for manual progress boosting via MPI_Test in applications where helper threads are either unavailable or undesirable.

Assess how asynchronous progress affects energy efficiency, core utilization, and thread scheduling on multi-core/many-core systems.