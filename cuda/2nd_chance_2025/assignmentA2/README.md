# Second Chance - 2025



## Key Capabilities

- ✅ ....


## Architecture Overview



## Assignment 1 - Row summatory of a matrix on a GPU



## Profiling

A GPU-accelerated implementation of row matrix summ using CUDA and OpenMP for the CPU validation, designed for performance analysis via Nsight Systems.

- NVIDIA A100

<img src="assignmentA1/profiling_logs/nsys_row_sum_a100.png" alt="NSight - Nsys row_sum rep a100" width="500">

- NVIDIA T4

<img src="assignmentA1/profiling_logs/nsys_row_sum_tesla.png" alt="NSight - Nsys row_sum report T4" width="500">

---

### Nsight GPU Profiling Summary (A100)

---

### GPU Specs: NVIDIA A100 vs T4

---


## Analysis
> **Note:** ... .



### CPU vs GPU Benchmark Results (NVIDIA A100)


### Speedup vs Matrix Size (T4 vs A100)


> **Note:** 
---

### Optimization Observations

- **A100 GPU shows significantly higher computational throughput**, ....

- **Memory operations remain a major bottleneck**, ....

- **GPU Occupancy**:
  - T4 GPU
    - Kernel execution time for large matrix (20480x20480) is approximately **25 ms**.

  - A100 GPU
    - Kernel execution is significantly faster (~**11–14 ms** for large matrix).
        ....

- **Recommendations for both GPUs**:
  - Use **`cudaMemcpyAsync` + streams** to 

- **CPU side is already optimized with OpenMP**:
  - Speedups are measured against a parallelized CPU implementation.
  - The GPU still outperforms CPU even with multithreaded loops, especially on larger workloads.



---


## CUDA & HPC Reference Materials

### CUDA Programming Guides & Core Concepts
- [CUDA C++ Programming Guide – Function Qualifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-qualifiers)
- [CUDA C++ Programming Guide – Tiled Partitions & Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tiled-partitions-cg)
- [Cooperative Groups – NVIDIA Developer Blog](https://developer.nvidia.com/blog/cooperative-groups/)
- [Robust and Scalable CUDA with Cooperative Groups (PDF)](https://leimao.github.io/downloads/blog/2024-08-06-CUDA-Cooperative-Groups/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)

### CUDA Code Examples & Bootcamps
- [CUDA Examples by Dr. Ken Netz](https://github.com/drkennetz/cuda_examples/)
- [CUDA Practice by A. Hamdi](https://github.com/a-hamdi/GPU/tree/main)
- [GPU Bootcamp Resources – OpenHackathons](https://github.com/openhackathons-org/gpubootcamp.git)

### Research Papers & Technical Reports
- [mergeSort(): A Graphical, Recursive, C++ Explanation](https://www.youtube.com/watch?v=RZK6KVpaT3I)
- [mergeSort(): CPU based reference](https://www.geeksforgeeks.org/merge-sort/)

### Cluster & System Management
- [NVIDIA Bright Cluster Manager 9.2 – Admin Manual (PDF)](https://support.brightcomputing.com/manuals/9.2/admin-manual.pdf)

### Educational Resource
- [Coding the Matrix – Linear Algebra for Computer Science](https://codingthematrix.com/)
