# Optimization Strategies for `sumvalues.c` and Derivative Versions

## 1. Dynamic Process Management (DPM) and MPI Initialization

The optimization journey of `sumvalues.c` explored **MPI Dynamic Process Management (DPM)**, using `MPI_Comm_spawn`
based on the MPI 3.1 Standard (Section 10). This allowed dynamic allocation of child processes based on the
input workload size (rows × columns), enabling runtime flexibility in resource use.

- **Advantages**:
  - Adaptability to runtime workload size.
  - Clear separation between parent (data reader) and child (worker) processes.
  
- **Challenges**:
  - Requires MPI implementation and fabric support (e.g., `I_MPI_SPAWN=on` for Intel MPI).
  - Less portable in tightly managed clusters (e.g., SLURM).

A comparison with static `mpirun -n N` launches showed that for most use cases, a singleton `MPI_Init` with
runtime control logic (without spawning) is simpler, faster, and more portable — unless adaptive or recursive
spawning is truly necessary.

---

## 2. Shared Memory and MPI I/O Improvements

- **Shared Memory**:
  - Deferred use of `MPI_Comm_split_type` for intra-node shared memory optimizations.
  - Suitable mainly when memory access dominates over I/O bottlenecks.

- **MPI I/O**:
  - Implemented `MPI_File_read_all` and `MPI_Type_create_subarray` in `sumvalues_mpio.c` and `sumvalues_mpio_dyn.c`.
  - Provided collective, efficient, and deterministic data loading from large input files.
  - Showed strong benefits for large-scale matrices (≥ 1000 × 1000).

For small matrices, standard file reading is more practical due to reduced complexity.

---

## 3. Key Feature Comparison & Usefulness

| Version                | Key Feature                                              | Investigation Worthiness |
|------------------------|----------------------------------------------------------|---------------------------|
| `sumvalues.c`          | Static MPI, dynamic workload-based process sizing        | ✅ Baseline & portable    |
| `sumvalues_spawn.c`    | Runtime `MPI_Comm_spawn` dynamic process creation        | ⚠️ Powerful but less robust |
| `sumvalues_mpio.c`     | MPI I/O with subarrays and fixed size setup              | ✅ Efficient for large data |
| `sumvalues_mpio_dyn.c` | MPI I/O + dynamic process sizing                         | ✅ Best hybrid approach   |

---

## 4. Conclusion

Optimizing `sumvalues.c` across multiple versions has demonstrated effective use of advanced MPI 3.1 features.
**Best practice**: use static `MPI_Init` + dynamic workload partitioning for most HPC workloads, and integrate
MPI I/O for scalable data access. **Use `MPI_Comm_spawn` only when runtime process adaptation is essential**.


## 5. Future Work

Both strategies worth it to keep exploring to improve the application from such perspectives:


- Exploring MPI shared memory windows (`MPI_Win`).
- Integrating hybrid MPI + OpenMP for NUMA-aware optimization.


- **sumvalues_mpio.c**
  - Static Parallel I/O with MPI Derived Datatypes — Implements parallel file access using MPI_Type_create_subarray and MPI_File_read_all for consistent and efficient workload distribution across fixed MPI ranks with collective I/O.

```bash
  mpicc -fopenmp -Wall -Wextra -O3 -march=native -funroll-loops -ffast-math -finline-functions -ftree-vectorize -fopt-info-vec-optimized sumvalues_mpio.c -o sumvalues_mpio -lm
  
  mpirun -n 4 ./sumvalues_mpio numbers1M 100 100
  mpirun -n 8 ./sumvalues_mpio numbers1M 4000 200
```

- **sumvalues_mpio_dyn.c**
  - Adaptive Parallel I/O Scaling — Enhances sumvalues_mpio.c with dynamic selection of the number of MPI ranks based on problem size, while preserving collective I/O through MPI_File interfaces.

```bash
  mpicc -fopenmp -Wall -Wextra -O3 -march=native -funroll-loops -ffast-math -finline-functions -ftree-vectorize -fopt-info-vec-optimized sumvalues_mpio_dyn.c -o sumvalues_mpio_dyn -lm
  
  mpirun -n 4 ./sumvalues_mpio_dyn numbers1M 100 100
  mpirun -n 8 ./sumvalues_mpio_dyn numbers1M 4000 200
```

- **sumvalues_spawn.c**
  - Dynamic Process Creation with MPI_Comm_spawn — Demonstrates true runtime spawning of child processes depending on matrix size, enabling flexible job scaling without needing mpirun -n configuration. Highlights the use of intercommunicators and manual data distribution from parent to children.
  - Parent-Child Task Orchestration with Explicit Messaging — The parent rank handles data loading and segmentation, explicitly dispatching submatrices and receiving computed results using point-to-point communication, making it robust for modular and on-demand workloads.

```bash
  mpicc -fopenmp -Wall -Wextra -O3 -march=native -funroll-loops -ffast-math -finline-functions -ftree-vectorize -fopt-info-vec-optimized sumvalues_spawn.c -o sumvalues_spawn -lm
  
  mpirun -n 1 ./sumvalues_spawn numbers1M 100 100
```



## 6. References 
- [Parallel and High Performance Computing](www.manning.com/books/parallel-and-high-performance-computing)
- [Introduction to the Message Passing Interface (MPI) - University of Stuttgart; High-Performance Computing-Center Stuttgart HLRS](https://fs.hlrs.de/projects/par/par_prog_ws/pdf/mpi_3.1_rab.pdf)
