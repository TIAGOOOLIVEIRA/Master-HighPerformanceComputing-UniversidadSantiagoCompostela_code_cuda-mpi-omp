# HPC Master’s Degree Repository

Welcome to my repository dedicated to tasks, assignments, and investigations related to my Master’s Degree in **High-Performance Computing (HPC)**. This repository serves as a structured collection of source code, scripts, tools, and reference materials developed or studied throughout the course.

---

## 🗂️ Repository Structure

This repository is organized into the following main directories:

1. **`cuda/`**  
   Contains assignments and projects completed as part of the coursework. Examples include:
   - **CUDA Applications**: Parallelized algorithms using NVIDIA CUDA inC/C++.
   - **Jetson Nano**: A lab to deploy a docker container with CUDA libraries in a Jetson Nano device so to perform fine-tuning and inferences for Computing Vision.
   
2. **`cloud/`**  
   Scripts and configurations for setting up and managing HPC workloads on cloud platforms, particularly **AWS**. This includes:
   - Job submission scripts for AWS Batch.
   - Auto-scaling configurations for HPC clusters leveraging **Cloud Formation**.
   - Optimization for cloud-based compute nodes.

3. **`hpc_tools/`**  
   Tools and utilities for profiling, debugging, and analyzing performance:
   - Usage of **NVIDIA Nsight**, **gprof**, **Valgrind**, and others.
   - **SpMV benchmarking**: Comparing compiler GCC and ICC performance improvements given optimization arguments also matrix sparsity strategies (Dense, COO, CSR, CSC)
   - Custom scripts for automating performance analysis workflows.
   - **PyTorch**: To evaluate PyTorch fine-tunning/training and inferences in a distributed architecture leveraging Lightning and other strategies for distributing workload.
   - **Ray** To evaluate how PyTorch atop Ray behaves.

4. **`mpi/`**  
   Exercises, code samples, and notes based on:
   - **Distribution stratgies**: Reduce/scatter, all gather, etc.
   - Advanced concepts for workload distribution and local parallelism leveraging OMP.

5. **`omp/`**  
   Exercises, code samples, and notes based on:
   - **Challenge tasks**: To implement and profiling algorithms like KMeans, KNN and Seismic leverraging high paralellism in OpenMP.
   - **Profiling tools**: Leverage tools like LIKWID and gprof

6. **`opencl/`**  
   Documentation, code samples, and notes based on:
   - **Books**: Summaries and examples from key HPC textbooks.
   - **Papers**: Reproducing experiments or methodologies from academic research.
   - **Trainings**: Exercises and examples from HPC-related training courses.
  
7. **`advanced_parallelprogramming/`**
   Source code for exercises in C++ utilizing advanced parallelism and workload distribution capabilities in an HPC environment:
   - **Threads**: 
   - **Mutex**: 
   - **Atomic**: 
   - **Threading Building Blocks**:
   - - **Load Balancing**: 

8. **`fpga/`**
   Source code for exercises in C++ utilizing advanced parallelism and workload distribution capabilities in an HPC environment:
   - **Threads**: 


---

## 💡 Objectives

This repository aims to:
- **Demonstrate hands-on knowledge** in programming for parallel and distributed systems.
- **Document the learning journey** in HPC concepts and their practical applications.
- **Facilitate sharing** of insights, examples, and best practices in HPC development.

---

## 🔧 Tools & Technologies

The repository includes work involving the following tools and technologies:

- **Programming Languages**:  
  Primarily C and C++ with extensions for CUDA, OpenMP, and MPI.

- **HPC Tools**:  
  - Profilers: NVIDIA Nsight, gprof.  
  - Debuggers: Valgrind, CUDA Debugger.  
  - Performance Analyzers: Perf, VTune.

- **Cloud Platforms**:  
  - AWS for on-demand HPC resources.  
  - Scripts for auto-scaling and cost-efficient usage.

---

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://www.usc.gal/en/studies/masters/engineering-and-architecture/master-high-performance-computing-online.git


2. Navigate to the appropriate folder for the task or topic you are interested in:
   ```bash
   cd hpc-masters/assignments

3. Follow the instructions or README files within each folder to build, execute, or explore the materials.

## 📚 Resources & References

The following resources have been instrumental in this journey:

    Books: "Parallel and High Performance Computing; C++ Concurrency in Action; " and others.
    Trainings: NVIDIA Deep Learning Institute, AWS HPC Workshops; HLRS
    Papers: ACM Research articles on performance analysis, scaling HPC applications, and optimization strategies.
