High-Performance Computing (HPC), also known as accelerated computing, aggregates the computing power from a cluster of nodes to divide and conquer complex tasks, achieving significantly higher performance than a single machine. It is essential for processing the massive amounts of data generated today and for solving intricate scientific and engineering problems, such as drug discovery, flight simulations, and financial risk analysis

**Limitations of On-Premises HPC**

Traditionally, HPC applications have faced several limitations when run on-premises [wilson2016experiences]:

- High upfront capital investment and long procurement cycles
- Challenges in maintaining infrastructure over its life cycle and managing technology refreshes
- Difficulty in forecasting annual budget and capacity requirements
- Limited scalability and elasticity, as specialized hardware like GPUs and serverless technologies are not readily available and require significant procurement and maintenance efforts
- Inability to efficiently cater to diverse HPC application needs such as parallel processing, low-latency/high-throughput networking, and fast I/O subsystems, leading to reduced efficiency and lost opportunities
- Traditional HPC systems are compute-centric, emphasizing floating-point performance, but scientific applications are increasingly data-intensive, posing challenges for existing architectures
- Voluminous data can consume network bandwidth and cause traffic issues during transfer, and local data centers struggle with data access, I/O, backup, power, and cooling [@mandoiu2016computational]

**Benefits of HPC in the Cloud**

Moving HPC workloads to the cloud offers significant advantages over on-premises solutions, effectively overcoming many of these limitations [@khanuja2022applied]:

- Virtually unlimited capacity: The cloud provides access to virtually unlimited HPC capacity, enabling users to move beyond fixed infrastructure constraints
- Drives innovation: It breaks barriers to innovation by allowing users to rapidly experiment with new approaches and make data-driven decisions
- This eliminates the need for rework and hardware maintenance, letting teams focus on business use cases. The elasticity of the cloud allows infrastructure to scale up or down based on demand
- Optimizes performance: Cloud HPC enables efficient resource utilization and supports rapid benchmarking and load testing, helping to optimize workloads without worrying about the cost, as you only pay for the resources used
- Cloud platforms provide compute, storage, and networking services specifically designed for HPC, eliminating long procurement cycles for specialized hardware
- Amplifies operational efficiency: Cloud platforms allow for process automation, frequent and reversible changes, and continuous improvement, supporting the development and execution of workloads efficiently
- On-demand access to compute capacity minimizes job queues, allowing teams to focus on critical problems
- Optimizes cost: The pay-per-use model significantly reduces high upfront capital investments
- Leveraging services like Amazon EC2 Spot Instances can lead to savings of up to 90% compared to on-demand instances for containerized workloads.
- Enables secure collaboration: Cloud platforms provide a collaborative environment for distributed teams to interact with simulation models in near real-time, ensuring security and compliance without physically moving data

**Industry-Specific Needs: Science**

HPC in the cloud is driving innovation across various industries, particularly in science. Life Sciences and Healthcare (Genomics, Computational Chemistry & Drug Design):

- HPC technology in the cloud allows the analysis of massive amounts of sensitive genomic data to gain insights into critical diseases, significantly reducing time for lab sample testing and drug discovery, while meeting security and compliance requirements
- For example, Novartis used the AWS cloud to screen 10 million compounds against a cancer target in less than a week, taking only 9 hours and costing $4,232. This experiment would have required an estimated $40 million investment and much longer in-house
- Challenges like large genomic datasets with issues in discoverability, accessibility, availability, and scalable processing are addressed by cloud storage systems like Amazon S3 (for virtually unlimited, durable storage) and AWS DataSync (for secure transfer) [@fernandezfraga2024applying]


**Addressing Broader Computational Challenges** 

Cloud HPC platforms and related technologies address fundamental computational challenges:

- Heterogeneous Computing: Modern platforms integrate multi-core CPUs with GPUs and FPGAs
- FPGAs can implement specialized computer architectures by configuring logic elements, dramatically accelerating algorithms like the Smith-Waterman by up to 160-fold compared to software implementations. SYstem-wide Compute Language (SYCL) as well offers a robust ecosystem for heterogeneous C++ compute acceleration across various hardware platforms and vendors. Apple Silicon M-Series SoCs (M1, M2, M3, M4) integrate CPU, GPU, memory, storage controllers, and specialized accelerators like Advanced Matrix Extensions (AMX) and Neural Engine into a single chip, providing high memory bandwidth and energy-efficient HPC potential

**Accelerated compute instances on AWS**
On AWS, P4d instances can be used to provision a supercomputer or an EC2 Ultracluster with more
than 4,000 A100 GPUs, Petabit-scale networking, and scalable, shared high throughput storage on
Amazon FSx for Lustre. Application and package developers use the NVIDIA CUDA library to build massively parallel applications for HPC and deep
learning [@khanuja2022applied].
