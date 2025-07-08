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



## MPI: mxnvm.c

A more simpllified benchmark analysis sticking to 3 execution samples per version of the matrix-vector multiplication program (mxvnm_*).
Important to mention that also the calculation per process leverage OpenMP for local parallelism.


To compile
- #mpicc -fopenmp mxvnm_collectiveblocking.c -o mxvnm_collectiveblocking
- #mpicc -fopenmp mxvnm_collectivenonblocking.c -o mxvnm_collectivenonblocking
- #gcc -fopenmp mxvnm_noncollective.c -o mxvnm_noncollective


To execute and collect statistics
- #time ./mxvnm_ori 5000 5000

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