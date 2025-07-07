# Labs 2.- MPI: Nonblocking Collective Communications


## MPI: pi_integral.c

After implementing the MPI nonblocking collective operations for pi_integral.c the shell from Lab1 pi_integral.c was adapted so the same analysis on the execution could be done to perform the comparison between both approaches.

To compile
- #gcc -c ../../Lab1/shared/place_report_mpi.c -o place_report_mpi.o
- #mpicc pi_integral.c ../../Lab1/shared/place_report_mpi.c -o pi_integral -I ../../Lab1/shared -lpthread

For the interactive mode, the pi_integral.sh can be changed its access mode as follows
- #chmod +x pi_integral.sh
- #./pi_integral.sh

Otherwise it is just the matter to submit the file as a job to the slurm scheduler
- #sbatch pi_integral.sh

The speedup analysis for the pi_integral.c in the Lab1 is taken into account to perform the analysis comparison as follows.


### MPI+OpenMP Speedup Table – Nonblocking Collectives

| MPI × OMP | Total Cores | Time (s) | Speedup vs Baseline | Observations                         |
|-----------|-------------|----------|----------------------|--------------------------------------|
| 1 × 1     | 1           | 6.200    | 1.00× (baseline)     | 🔵 Serial, blocking version baseline |
| 2 × 8     | 16          | 0.056433 | 109.90×              | 🟢 Superb hybrid scaling             |
| 4 × 4     | 16          | 0.049028 | 126.46×              | 🟢 Best hybrid balance (nonblocking) |
| 8 × 2     | 16          | 0.050750 | 122.22×              | 🟢 Strong hybrid, low sync overhead  |
| 16 × 1    | 16          | 0.048685 | 127.37×              | 🟢 All-MPI shines with nonblocking   |


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
- #mpicc dotprod.c ../../Lab1/shared/place_report_mpi.c -o dotprod -I ../../Lab1/shared -lpthread

For the interactive mode, the pi_integral.sh can be changed its access mode as follows
- #chmod +x dotprod.sh
- #./dotprod.sh

Otherwise it is just the matter to submit the file as a job to the slurm scheduler
- #sbatch dotprod.sh


The speedup analysis for the dotprod.c in the Lab1 is taken into account to perform the analysis comparison as follows.


### MPI+OpenMP Speedup Table – Nonblocking Collectives



### Conclusions