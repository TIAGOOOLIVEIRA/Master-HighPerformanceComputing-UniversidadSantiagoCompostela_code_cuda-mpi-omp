# Second Chance - 2025


## Assignment 1 - Row summatory of a matrix on a GPU

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| 2 | 3 | 4 | 5 | 6 |
| 3 | 4 | 5 | 6 | 7 |
| 4 | 5 | 6 | 7 | 8 |
| 5 | 6 | 7 | 8 | 9 |
                ↓ Row Sum
                
              [ 15 | 20 | 25 | 30 | 35 ]



## Profiling

A GPU-accelerated implementation of row matrix summ using CUDA and OpenMP for the CPU validation, designed for performance analysis via Nsight Systems.

<img src="profiling_logs/nsys_row_sum_a100.png" alt="NSight - Nsys row_sum rep a100" width="500">
<img src="profiling_logs/nsys_row_sum_t4.png" alt="NSight - Nsys row_sum report T4" width="500">

---

### 📊 Nsight GPU Profiling Summary (A100)

| Component              | Time (ms)   | % Total CUDA API Time | Notes                                        |
|------------------------|-------------|------------------------|----------------------------------------------|
| `cudaMemcpy (HtoD)`    | 155.738     | 27.0%                  | Host to Device transfer – still dominant     |
| `row_sum_kernel`       | 11.470      | 2.0%                   | Kernel execution on A100 – very fast!        |
| `cudaMalloc`           | 576.419     | 58.4%                  | Likely includes overhead of memory tracking  |
| `cudaFree`             | 0.982 µs    | ~0%                    | Negligible                                   |
| `cudaEventRecord`      | 232.982 µs  | ~0%                    | Event markers for timing                     |
| `cudaEventSynchronize` | 11.469 ms   | ~2.0%                  | Synchronization time                         |
| `cudaLaunchKernel`     | 5.645 ms    | ~1.2%                  | Kernel launch cost                           |

---

### GPU Specs: NVIDIA A100 vs T4

| Property               | Tesla T4             | A100 (40GB)           |
|------------------------|----------------------|------------------------|
| Architecture           | Turing (SM 7.5)      | Ampere (SM 8.0 / 8.6)  |
| CUDA Cores             | 2,560                | 6,912                  |
| Memory Size            | 16 GB GDDR6          | 40 GB HBM2e            |
| Memory Bandwidth       | ~320 GB/s            | ~1.6 TB/s              |
| L2 Cache               | 4 MB                 | 40 MB                  |
| Compute Capability     | 7.5                  | 8.0 / 8.6              |
| Tensor Cores           | Yes (limited)        | Yes (3rd Gen)          |
| Max Threads/Block      | 1024                 | 1024                   |

---


## Analysis
> **Note:** The speedup values reported here are measured **against a CPU baseline using OpenMP multithreading**, not serial execution. This means the GPU performance is being compared to an already-optimized parallel CPU implementation running with up to 8 threads. The observed speedups therefore highlight the additional parallelism and memory throughput advantages offered by the GPU.



### CPU vs GPU Benchmark Results (NVIDIA A100)

| Matrix Size   | Threads/Block | GPU Time (ms) | CPU Time (ms) | Speedup (CPU / GPU) | Validation |
|---------------|----------------|----------------|----------------|----------------------|------------|
| 5376 x 5376   | 32             | 4.4996         | 12.4370        | 2.76×                | ✅ Passed  |
| 5376 x 5376   | 64             | 4.7641         | 9.6350         | 2.02×                | ✅ Passed  |
| 5376 x 5376   | 128            | 4.7580         | 13.9200        | 2.93×                | ✅ Passed  |
| 10880 x 10880 | 32             | 18.7215        | 50.3350        | 2.69×                | ✅ Passed  |
| 10880 x 10880 | 64             | 7.9227         | 55.6190        | 7.02×                | ✅ Passed  |
| 10880 x 10880 | 128            | 7.7418         | 55.3250        | 7.15×                | ✅ Passed  |
| 20480 x 20480 | 32             | 12.9756        | 195.2110       | 15.04×               | ✅ Passed  |
| 20480 x 20480 | 64             | 18.5411        | 191.9850       | 10.36×               | ✅ Passed  |
| 20480 x 20480 | 128            | 14.3911        | 190.3800       | 13.23×               | ✅ Passed  |


### Speedup vs Matrix Size (T4 vs A100)


> **Note:** Speedups are computed relative to a multithreaded CPU implementation using OpenMP (not a serial CPU baseline).
This table helps highlight not only how both GPUs outperform the CPU, but also how A100 scales more efficiently on large workloads — especially for the 20480 x 20480 matrix.

| Matrix Size   | Threads/Block | CPU Time (ms) | GPU Time (T4, ms) | Speedup (T4) | GPU Time (A100, ms) | Speedup (A100) |
|---------------|----------------|----------------|--------------------|---------------|----------------------|----------------|
| 5376 x 5376   | 32             | 11.2690        | 5.2995             | 2.13×         | 4.4996               | 2.50×          |
| 5376 x 5376   | 64             | 11.2640        | 5.2599             | 2.14×         | 4.7641               | 2.36×          |
| 5376 x 5376   | 128            | 11.2860        | 5.1036             | 2.21×         | 4.7580               | 2.37×          |
| 10880 x 10880 | 32             | 46.1170        | 8.7408             | 5.27×         | 18.7215              | 2.46×          |
| 10880 x 10880 | 64             | 46.1060        | 8.8149             | 5.23×         | 7.9227               | 5.82×          |
| 10880 x 10880 | 128            | 46.1280        | 11.3081            | 4.08×         | 7.7418               | 5.96×          |
| 20480 x 20480 | 32             | 164.0430       | 26.5357            | 6.18×         | 12.9756              | 15.04×         |
| 20480 x 20480 | 64             | 164.3190       | 23.3596            | 7.03×         | 18.5411              | 10.34×         |
| 20480 x 20480 | 128            | 164.2960       | 25.3311            | 6.49×         | 14.3911              | 11.41×         |

---

### Optimization Observations

- **A100 GPU shows significantly higher computational throughput**, especially for large matrices. At 20480x20480, it delivers over **11× speedup vs CPU**, and nearly **2× better performance** than the T4 for the same workload.

- **Memory operations remain a major bottleneck**, especially `cudaMemcpy` (host-to-device), which can account for over 80% of the total runtime depending on matrix size.

- **A100 is more efficient with larger workloads**:
  - On smaller matrices, T4 and A100 perform similarly.
  - On larger matrices, A100 shows much better scaling, due to:
    - Higher memory bandwidth (~1.6 TB/s vs T4’s 320 GB/s)
    - Larger L2 cache and faster HBM2e memory
    - More CUDA cores (6912 vs 2560)
    - Better SM occupancy and instruction throughput

- **Observed issue on A100**:
  - `cudaMalloc` had unusually high duration in some runs (up to 576 ms).
  - This may be due to internal driver-side memory initialization.
  - Consider switching to **memory pools** (`cudaMallocAsync` with `cudaMemPool_t`) to avoid repeated allocations.

- **GPU Occupancy**:
  - T4 GPU
    - Kernel execution time for large matrix (20480x20480) is approximately **25 ms**.
    - Most of the time is spent in **`cudaMemcpy` (host-to-device)**, not in kernel compute.
    - Occupancy is **likely moderate**, but not the main limiting factor.
    - Potential constraints:
    - Limited shared memory or register usage
    - Lower memory bandwidth (~320 GB/s)

  - A100 GPU
    - Kernel execution is significantly faster (~**11–14 ms** for large matrix).
    - With faster SMs, more shared memory, and more registers, **A100 maintains higher occupancy** and throughput.
    - Efficient instruction pipelines allow A100 to scale better on large workloads.
    - Launch overhead and compute time are well balanced — a sign of good **warp-level efficiency**.

- **Recommendations for both GPUs**:
  - Use **`cudaMemcpyAsync` + streams** to overlap transfers with computation.
  - Consider **`cudaMallocManaged`** and **`cudaMemPrefetchAsync`** for simpler and more efficient memory usage.
  - Avoid redundant memory allocations inside loops.
  - Profile with **Nsight Compute** to dig into:
    - Warp efficiency
    - Shared memory usage
    - Cache hit rates (L1/L2/global memory)
    - SM occupancy metrics

- **CPU side is already optimized with OpenMP**:
  - Speedups are measured against a parallelized CPU implementation.
  - The GPU still outperforms CPU even with multithreaded loops, especially on larger workloads.



---


## References
```C
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-qualifiers
https://github.com/drkennetz/cuda_examples/
https://github.com/a-hamdi/GPU/tree/main
https://github.com/openhackathons-org/gpubootcamp.git
https://developer.nvidia.com/blog/cooperative-groups/
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tiled-partitions-cg
Hardware Compute Partitioning on NVIDIA GPUs* https://www.cs.unc.edu/~jbakita/rtas23.pdf
CUDA and Applications to Task-based Programming https://diglib.eg.org/server/api/core/bitstreams/3e283a2e-e6a3-4908-8d77-1741d01cc06f/content
NVIDIA Bright Cluster Manager 9.2 https://support.brightcomputing.com/manuals/9.2/admin-manual.pdf
COOPERATIVE GROUPS https://leimao.github.io/downloads/blog/2024-08-06-CUDA-Cooperative-Groups/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
https://leimao.github.io/downloads/blog/2024-08-06-CUDA-Cooperative-Groups/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf
https://codingthematrix.com/
```