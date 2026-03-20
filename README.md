# HPC Master’s Degree Repository

Welcome to my repository dedicated to assignments, experiments, and investigations developed throughout my Master’s Degree in **High-Performance Computing (HPC)**.

This repository consolidates source code, performance studies, cloud experiments, distributed training evaluations, and acceleration strategies explored during the program.

---

## 🗂️ Repository Structure

The repository is organized into the following main directories (alphabetical order):

### `advanced_parallelprogramming/`
C++ exercises focused on advanced concurrency and workload distribution:
- Multithreading and synchronization (mutexes, atomics)
- Task-based parallelism (e.g., Threading Building Blocks)
- Load balancing strategies
- Performance-oriented concurrent design patterns

---

### `cloud/`
Experiments and scripts for running HPC and distributed workloads in the cloud (primarily AWS):
- HPC cluster provisioning (CloudFormation-based setups)
- Job submission workflows
- Auto-scaling configurations
- Cost-efficiency and resource optimization studies

---

### `cuda/`
CUDA-based GPU acceleration projects:
- Parallel algorithm implementations in CUDA C/C++
- Performance optimization strategies
- Jetson Nano lab: Dockerized CUDA environment for computer vision fine-tuning and inference

---

### `fpga/`
High-Level Synthesis (Vitis HLS) experiments targeting hardware acceleration:
- 2D convolution optimization
- Exploration of FPGA acceleration strategies in HPC contexts

---

### `hpc_tools/`
Profiling, benchmarking, and distributed AI evaluation tools:
- Profiling with NVIDIA Nsight, gprof, Valgrind, LIKWID
- SpMV benchmarking (Dense, COO, CSR, CSC) with GCC and ICC optimization comparisons
- Distributed PyTorch training (Lightning, DDP strategies)
- Evaluation of PyTorch on top of Ray for distributed workloads
- Automated performance analysis scripts

---

### `mpi/`
Distributed-memory programming exercises:
- Collective communication patterns (reduce, scatter, all-gather, etc.)
- Hybrid strategies combining MPI with OpenMP
- Workload distribution techniques

---

### `omp/`
Shared-memory parallel programming using OpenMP:
- Parallel implementations of algorithms (KMeans, KNN, Seismic workloads)
- Profiling and performance tuning
- Scaling and scheduling strategies

---

### `opencl/`
Cross-platform acceleration experiments:
- Portable kernel development
- Vector-based kernel execution
- Multi-platform execution (Linux HPC environments and macOS)

---

## 💡 Objectives

This repository aims to:

- Demonstrate practical expertise in parallel and distributed programming.
- Explore performance optimization across CPUs, GPUs, FPGAs, and cloud environments.
- Evaluate scalability, workload distribution, and cost-performance trade-offs.
- Document applied research in HPC, distributed AI, and scientific computing workflows.

---

## 🔧 Tools & Technologies

### Programming Languages
- C / C++
- CUDA C++
- OpenMP
- MPI
- OpenCL
- Python (for distributed AI and orchestration)

### Performance & Profiling Tools
- NVIDIA Nsight
- gprof
- Valgrind
- LIKWID
- Intel VTune
- Linux perf

### Distributed & AI Frameworks
- PyTorch
- PyTorch Lightning
- Ray

### Cloud & Infrastructure
- AWS (HPC clusters, scaling strategies)
- Infrastructure as Code (CloudFormation)

---

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/TIAGOOOLIVEIRA/Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp.git
