# CUDA Merge Sort Report

## 1. Introduction

This work focuses on implementing a merge sort algorithm optimized for execution on NVIDIA GPUs using CUDA. The objective is to evaluate performance when using static and dynamic shared memory, particularly examining memory size limits and strategies to avoid bank conflicts. Arrays `arr`, `arr1`, and `arr2` are allocated in shared memory to enable fast intra-block data exchange. The merge function is parallelized across threads with support for atomic operations to synchronize access.

The algorithm processes 32,768 arrays of sizes ranging from 8 to 1024 (powers of two) on T4, and up to 16,384 on A100 for dynamic memory. Profiling tools such as Nsight Systems, Nsight Compute, and Intel VTune were critical in identifying bottlenecks and validating optimizations. Workflow automation via shell scripts, SLURM job submissions for T4 and A100 GPU nodes, and Makefile compilation allowed reproducible and scalable experimentation.

---

## 2. How to

This project is structured into two folders:

- `mergesort_dynamic`: Contains the CUDA implementation using dynamic shared memory  
- `mergesort_static`: Contains the implementation with static shared memory allocation

Both folders include:

- `mergesort.cu`: Source code  
- `Makefile`: Handles compilation, profiling, reporting, and cleaning  
- `mergesort_job.sh`: Executes the program either in interactive mode (T4) or via SLURM (A100)  
- `shared/helper.cuh`: Common utilities for both versions

### Running Instructions:

```bash
cd mergesort_dynamic  # or mergesort_static
make                  # Compile
./mergesort [array_size] [dump_flag]
# Optional: Run job script
chmod +x mergesort_job.sh
./mergesort_job.sh
```

Defaults: `array_size = 1024`, `dump_flag = 0`

---

## 3. Benchmarking

### Dynamic Shared Memory Results


| Array Size | Avg GPU (T4) ms | Avg CPU (T4, OMP-8) ms | Speedup (T4) | Avg GPU (A100) ms | Avg CPU (A100, OMP-8) ms | Speedup (A100) | Observation                 |
|------------|------------------|--------------------------|--------------|-------------------|----------------------------|----------------|----------------------------|
| 128        | 379.47           | 32.55                   | 0.09x        | 192.08            | 32.82                     | 0.17x          | CPU faster                 |
| 256        | 453.78           | 48.39                   | 0.11x        | 197.20            | 48.07                     | 0.24x          | CPU faster                 |
| 512        | 466.73           | 75.38                   | 0.16x        | 214.98            | 75.23                     | 0.35x          | GPU starts catching up     |
| 1024       | 437.44           | 121.22                  | 0.28x        | 380.37            | 121.01                    | 0.32x          | More balanced, A100 better|
| 16384      | —                | —                       | —            | 1492.71           | 254.67                    | 0.17x          | Only executed on A100      |


### Static Shared Memory Results


| Array Size | Avg GPU (T4) ms | Avg CPU (T4, OMP-8) ms | Speedup (T4) | Avg GPU (A100) ms | Avg CPU (A100, OMP-8) ms | Speedup (A100) | Observation                       |
|------------|------------------|--------------------------|--------------|-------------------|----------------------------|----------------|----------------------------------|
| 128        | 174.33           | 32.82                   | 0.19x        | 174.33            | 32.82                     | 0.19x          | Slightly faster than dynamic     |
| 256        | 177.56           | 48.07                   | 0.27x        | 177.56            | 48.07                     | 0.27x          | Same for both GPUs              |
| 512        | 180.91           | 75.23                   | 0.42x        | 180.91            | 75.23                     | 0.42x          | Higher gain vs dynamic version   |
| 1024       | —                | —                       | —            | 189.74            | 121.01                    | 0.63x          | T4 cannot run due to mem limits  |
| 16384      | —                | —                       | —            | —                 | —                         | —              | Not supported due to mem cap     |


### Memory Capacity Observation Table

| Array Size | Static Shared (T4) | Static Shared (A100) | Dynamic Shared (T4) | Dynamic Shared (A100) |
|------------|---------------------|------------------------|----------------------|------------------------|
| 8 - 64     | ✅                   | ✅                     | ✅                   | ✅                     |
| 128 - 256  | ✅                   | ✅                     | ✅                   | ✅                     |
| 512        | ⚠️ (fragile)        | ✅                     | ✅                   | ✅                     |
| 1024       | ❌ (overflow)        | ⚠️ (limit-bound)      | ⚠️ (slow)            | ✅                     |
| 16384      | ❌                   | ❌                     | ❌                   | ✅                     |


### Key Discoveries

- Static shared memory offered **faster execution**, especially for small to mid-size arrays.
- Dynamic shared memory allowed **larger problem sizes** (e.g., 16,384) to run on A100.
- **T4 was constrained by shared memory capacity**, especially for static allocation.
- **Warp divergence, occupancy, and bank conflict mitigation** impacted performance, confirmed via Nsight and VTune.
- CPU performance was **competitive at lower sizes**, making GPU worthwhile only for scale.

---

## 4. Profiling

The shell script mergesort_job.sh also autmates the generation of the profiling reports.

For profiling manually:
```bash
cd mergesort_dynamic  # or mergesort_static
make profile          # for Nsight System report
make vtune-cpu        # for VTune analysis only for the CPU computation
make ncu              # for Nsight Compute report
make clean            # for cleaning up compilation and profiling output
```

### Tooling used includes Nsight Systems, Nsight Compute, and Intel VTune:

  - Nsight Systems: Helped trace kernel launches, overlapping memory ops, and kernel execution patterns.

  - Nsight Compute: Revealed warp efficiency, scheduler divergence, shared memory bank conflicts, and memory throughput.

  - Intel VTune: Analyzed CPU-side parallelism and threading bottlenecks when launching kernels or managing host data.

Screenshots were collected and categorized by memory type (static/dynamic) and GPU (T4/A100).


## 5. Environment
### Compilation

In the Makefile the GPU architecture is captured dynamically via the following instruction

```bash
GPU_ARCH := $(shell if command -v nvidia-smi >/dev/null 2>&1; then \
	nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | sed 's/\.//'; \
	else echo 75; fi)
```

In order to compile manually for A100 GPU card and T4, following the commands
```bash
For A100 GPU Card
nvcc -gencode arch=compute_80,code=sm_80 -O3 -lineinfo -Xptxas -v -Xcudafe "--display_error_number" -Xcompiler="-march=native -fopenmp" mergesort.cu -o mergesort

For T4 GPU Card
nvcc -gencode arch=compute_75,code=sm_75 -O3 -lineinfo -Xptxas -v -Xcudafe "--display_error_number" -Xcompiler="-march=native -fopenmp" mergesort.cu -o mergesort
```

### Architecture Info


| GPU     | SM Count | Shared Mem per Block | Max Threads/Block | Warp Size | Clock MHz |
|---------|----------|----------------------|--------------------|-----------|-----------|
| T4      | 40       | 48 KB                | 1024               | 32        | 5001      |
| A100    | 108      | 48 KB                | 1024               | 32        | 1215      |

## 6. Future Work

  - Apply loop unrolling and restrict pointers

  - Implement bank-conflict-free memory patterns

  - Mitigate warp divergence further

  - Optimize shared memory layout with padding

  - Integrate CUDA Streams and Events for async overlapping

  - Use cooperative groups and persistent kernels

  - Port to C++ for pipeline API and multi-GPU orchestration

  - Evaluate MPI + CUDA for distributed sorting

  - Consider real-time visualization and tracing via Nsight Visual Studio

  - Explore bulk memory copies and pinned host memory

## 7. Conclusions

  - Static shared memory is faster but not scalable for large arrays due to its fixed size.

  - Dynamic shared memory enables larger-scale sorting but with performance cost on smaller arrays.

  - A100 significantly outperformed T4 for dynamic memory.

  - Performance insights via profiling tools guided optimal thread configurations and revealed memory limitations.

  - Significant learning occurred around CUDA tuning, shared memory layout, and profiling best practices.

## 8. References

### CUDA Programming Guides & Core Concepts
- [CUDA C++ Programming Guide – Function Qualifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-qualifiers)
- [CUDA C++ Programming Guide – Tiled Partitions & Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tiled-partitions-cg)
- [Cooperative Groups – NVIDIA Developer Blog](https://developer.nvidia.com/blog/cooperative-groups/)
- [Robust and Scalable CUDA with Cooperative Groups (PDF)](https://leimao.github.io/downloads/blog/2024-08-06-CUDA-Cooperative-Groups/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)
- [CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)
- [Advanced Performance Optimization in CUDA](https://www.nvidia.com/ja-jp/on-demand/session/gtc24-s62192/?playlistId=playList-d59c3dc3-9e5a-404d-8725-4b567f4dfe77)
- [Verification of Producer-Consumer Synchronization in GPU Programs](https://cs.stanford.edu/people/sharmar/pubs/weft.pdf)
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Register Cache: Caching for Warp-Centric CUDA Programs](https://developer.nvidia.com/blog/register-cache-warp-cuda/)
- [Simplifying GPU Programming for HPC with NVIDIA Grace Hopper Superchip](https://developer.nvidia.com/blog/simplifying-gpu-programming-for-hpc-with-the-nvidia-grace-hopper-superchip/)
- [Optimize GPU Workloads for Graphics Applications with NVIDIA Nsight Graphics](https://developer.nvidia.com/blog/optimize-gpu-workloads-for-graphics-applications-with-nvidia-nsight-graphics/)
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access Dec 04, 2013 By Justin Luitjens](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- [Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)
- [Walkthrough: Debugging a CUDA Application](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Debugging_CUDA_Application.htm)

### CUDA Code Examples & Bootcamps
- [CUDA Examples by Dr. Ken Netz](https://github.com/drkennetz/cuda_examples/)
- [CUDA Practice by A. Hamdi](https://github.com/a-hamdi/GPU/tree/main)
- [GPU Bootcamp Resources – OpenHackathons](https://github.com/openhackathons-org/gpubootcamp.git)

### Research Papers & Technical Reports
- [Sorting with GPUs- A Survey 1709.02520v1](arxiv.org/pdf/1709.02520)
- [Multicore and GPU Programming, An Integrated Approach, chapter 6 "GPU Programming CUDA"](https://www.sciencedirect.com/book/9780128141205/multicore-and-gpu-programming)

 - [Little's Law to explain GPU throughput vs. latency - Parallel Read Error Correction for Big Genomic Datasets](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7397660)
- [mergeSort(): A Graphical, Recursive, C++ Explanation](https://www.youtube.com/watch?v=RZK6KVpaT3I)
- [mergeSort(): CPU based reference](https://www.geeksforgeeks.org/merge-sort/)

### Cluster & System Management
- [NVIDIA Bright Cluster Manager 9.2 – Admin Manual (PDF)](https://support.brightcomputing.com/manuals/9.2/admin-manual.pdf)

### Educational Resource
- [Coding the Matrix – Linear Algebra for Computer Science](https://codingthematrix.com/)