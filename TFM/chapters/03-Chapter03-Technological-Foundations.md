# Chapter 3 - Technological Foundations

This chapter presents the core hardware and software tools that underpin our
two-phase DNA sequencing pipeline. We begin by detailing the distributed compute
framework used for data preparation, then describe the high-performance
infrastructure driving our numerical simulations, and finally outline the cloud services
and automation patterns that bind these elements into a cohesive, reproducible
system.

## 3.1. Ray Framework for Distributed Preprocessing

At the heart of Phase 1 lies Ray, an open-source, Python-native distributed execution
engine that unifies AI/ML and ETL workloads under a single API [12].
Ray tech stack is well integrated in the AWS platform, as can be seen in the Figure
3, where it benefits from the high level of automation for deployment and easy
integration with platform services.

The main Ray components are the following:

- Ray Core introduces two primitives — stateless tasks and stateful actors —
which map naturally onto data-parallel and model-parallel stages of our
embedding pipeline [2].
- Ray Data builds on Apache Arrow to load, clean, and transform terabytes of
FASTQ files in parallel, overlapping I/O and compute to minimize idle time [12].
- Ray AIR (AI Runtime) bundles Ray Data with Ray Train (distributed model
training), Ray Tune (hyperparameter search), and Ray Serve (scalable inference), enabling end-to-end feature extraction and embedding generation—all orchestrated as a single Python application [14].


<img src="../images/Anyscale-Ray-Gen-AI-6.png" alt="Anyscale-Ray-Gen-AI-6" width="500">

**Figure 3**: Phase 1 - Ray Anyscale on AWS.


## 3.2. Hardware Accelerators and HPC Infrastructure

Phase 2 leverages specialized hardware to ingest the compact embeddings and
perform compute-intensive simulations.
Also, using AWS Parallel Computing Service (AWS PCS), which leverages
ParallelCluster as it shows Figure 4, in “deployment mode” brings several
advantages that align perfectly with our two‐phase architecture. As a fully managed,
pay-as-you-go service, PCS lets you spin up Slurm‐based HPC clusters without
upfront hardware investment — automatically billing only for the compute, storage,
and networking. Deployment is a simple point-and-click or CLI/SDK command, with
built-in observability, automated updates, and elastic scaling, so the focus becomes
the Ray preprocessing (Phase 1) and MPI/Slurm simulations (Phase 2) rather than
cluster operations. PCS’s tight integration with AWS ParallelCluster templates and
FSx for Lustre storage simplifies data handoff between the GPU-accelerated Ray workers and the Slurm jobs, enabling a seamless, end-to-end pipeline — fully
defined as code and elastically managed in the cloud.

The main components used in this phase are the following:

- GPUs & FPGAs: NVIDIA A100 GPUs (via AWS P4d instances) and custom
FPGA kernels deliver thousands of cores for dense matrix multiplies, sparse
graph traversals, and alignment algorithms like Smith–Waterman — achieving
up to 160× speedups over CPUs in selected cases [7].
- AWS ParallelCluster: This managed service provisions Slurm/MPI clusters
with Elastic Fabric Adapter (EFA) for low-latency interconnects and Amazon
FSx for Lustre for high-throughput shared storage [1]. A CloudFormation (or
CDK) template configures head nodes, compute fleets, and networking, then
bootstraps and compiles the code for large-scale DNA simulations.

<img src="../images/2024-aws-pcs-1-diagram.png" alt="2024 aws pcs" width="500">

**Figure 4**: Phase 2 - HPC cluster provisioned with AWS ParallelCluster/PCS.


## 3.3. Cloud Storage, Networking, and Data Flow

A resilient data backbone ensures seamless interoperability between Ray and HPC
stages:

- Amazon S3 & Lake Formation serve as the centralized data lake and
metadata catalog for raw reads and embedding outputs.
- AWS DataSync securely transfers on-premises archives into S3, while
Amazon EFS and FSx for Lustre mount that data for both Ray workers and
HPC nodes [7].
- Hybrid Patterns: For bursty, stateless tasks we route workloads to AWS
Lambda (serverless), and for long-running, stateful simulations we use EC2
instances — following the “Mashup” hybrid execution model to optimize cost
and performance [8].

## 3.4. Infrastructure as Code and Automation

To achieve full reproducibility and rapid iteration, we codify every component using
the following IaC and CI/CD services:

- CloudFormation / CDK templates define both the Ray autoscaler clusters and
the ParallelCluster environment.
- AWS CodePipeline & CodeBuild automate Docker image builds (with the
GPU-enabled Ray runtime), ECR pushes, and stack deployments whenever
code is merged.
- Monitoring & Scaling: CloudWatch metrics (e.g. pending Ray tasks, Slurm
queue depth) trigger Lambda-backed autoscaling policies, ensuring clusters
are the right size for workload demands and that cost efficiency is maintained.

Together, these technological foundations — spanning Ray’s distributed compute,
GPU/FPGAs, AWS HPC services, and robust IaC pipelines — enable a highly
interoperable, scalable solution for next-generation DNA sequencing and other
data‐intensive scientific workflows.


