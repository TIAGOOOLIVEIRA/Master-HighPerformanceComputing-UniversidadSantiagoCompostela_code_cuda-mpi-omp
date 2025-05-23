# Second Chance - 2025 - Assignment 4

This project implements a high-performance Row summatory of multiple matrices on a GPU using CUDA and OpenMP. It is designed for two purposes:

- **Academic exploration** of GPU parallelization, occupancy tuning, and heterogeneous memory management.
- **Internal benchmarking** of GPU performance across different NVIDIA architectures using real-world profiling workflows.

## Key Capabilities



---

## Architecture Overview

```text

```

## Assignment 4 - Row summatory of a matrices on a GPU


- **Requirements**:
  - Implement a CUDA version of the row summatory for multiple matrices that is executed on the GPU
  - Input: “m” matrices of “nxn” floats. Initialize them as you want
  - Output: matrix of “mxn” floats (or “m” vectors of “n” elements)
  - The program must work for variable number of matrices “m” and different values of “n”
  - The row summatory of every matrix must be performed in parallel in the GPU by one kernel (probably the one that you designed for Assignment 1)
  - Overlap the transfer of the input matrix for iteration “i+1” with the kernel of matrix “i”
  - Hint: avoid using too many streams or too much memory, only the necessary for the correct overlapping
  - Hint: develop a C version and compare the result of both versions to check that your CUDA code provides the correct results
  - You must only provide the CUDA code (deadline May 23th)



## Profiling

A GPU-accelerated implementation of row matrix summ using CUDA and OpenMP for the CPU validation, designed for performance analysis via Nsight Systems.

- NVIDIA A100


<img src="images/nsy-rowsum_matrices-a100-nsyght.png" alt="NSY, Nsight" width="500">


- NVIDIA T4



---

### Nsight GPU Profiling Summary (A100)


---

### GPU Specs: NVIDIA A100 vs T4


---


## Analysis



### CPU vs GPU Benchmark Results (NVIDIA A100)




### Optimization Observations


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
- **Hardware Compute Partitioning on NVIDIA GPUs**  
  [UNC RTAS '23](https://www.cs.unc.edu/~jbakita/rtas23.pdf)
- **CUDA and Applications to Task-Based Programming**  
  [Eurographics Digital Library](https://diglib.eg.org/server/api/core/bitstreams/3e283a2e-e6a3-4908-8d77-1741d01cc06f/content)

### Cluster & System Management
- [NVIDIA Bright Cluster Manager 9.2 – Admin Manual (PDF)](https://support.brightcomputing.com/manuals/9.2/admin-manual.pdf)

### Educational Resource
- [Coding the Matrix – Linear Algebra for Computer Science](https://codingthematrix.com/)
