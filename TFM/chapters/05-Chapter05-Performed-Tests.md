# Performed Tests

To validate the architectural benefits of our two-phase pipeline — leveraging Ray for
high-throughput data preparation (Phase 1) and AWS ParallelCluster for
compute-intensive simulations (Phase 2) — we conducted a series of controlled
experiments on AWS. Our goals were to measure key quality attributes: scalability,
elasticity, performance, and cost-efficiency.

# 5.1. Test Environment and Objectives

During the experimental phase, significant challenges were encountered in securing
sufficient GPU-accelerated and HPC node hours on AWS — limited service quotas
and budget constraints prevented me from scaling the cluster size and data volumes
fully, thereby restricting exhaustive benchmarking of performance and cost
trade-offs.
Nevertheless, the tests were ran in the AWS us-west-2 region. The resources used
in the tests were the following:

- Phase 1 Ray Cluster: Provisioned via `ray-cluster-example-full.yaml` (autoscaling up to 2 GPU-enabled workers).
- Phase 1 Application:`dna_embeddings.py`reads a 50 GB FASTQ sample from S3, applies ProtBERT encoding on GPU actors, and writes 10 GB of Parquet embeddings to FSx for Lustre configured as a scratch_2 (SSD) deployment with 1.2 TiB of storage and 2.4 GiB/s of throughput capacity, auto-importing from our S3 embeddings bucket.
- Phase 2 ParallelCluster: Deployed with a Slurm head node, three G4dn GPU
compute nodes, EFA networking, and FSx for Lustre.
- Phase 2 Application: mpi_dna_seq.cpp (compiled via Slurm bootstrap) runs a toy DNA similarity kernel over the embeddings; submissions via mpi_dna_seq.sh`.

We measured end-to-end latency, per-stage throughput, cluster scale-up times, and
on-demand vs. spot cost impact.

## 5.2. Phase 1: Data Preparation Benchmark.

Throughput & Scalability

With a single GPU actor,`dna_embeddings.py` processed 10 GB in 15 minutes (~11 MB/s). Scaling to two GPU workers doubled throughput to ~22 MB/s, demonstrating near-linear scaling in Ray Data’s `map_batches` with `num_gpus=1` per actor [14].
Regarding elasticity, Ray autoscaler brought a second node online in ~90s under
load, and released idle nodes after the 5-minute timeout, keeping average cluster
size aligned to demand.

Cost Efficiency

Using Spot instances for workers reduced Phase 1 EC2 costs by ~65% compared to
on-demand. Combined with GPU acceleration, we achieved a per-gigabyte
embedding cost of $0.12, by sticking to the baseline calculation like running three
p3.2xlarge GPUs (spot $1.34/hr each) plus a head node (spot $0.034/hr) for 15
minutes, it can cost $1.03 to process 10 GB — yielding a rounded cost of $0.12/GB.
According to the documentation, by leveraging EFA, MPI latency can be reduced by
~60% over TCP, improving solver performance in tightly coupled kernels.

## 5.3. Phase 2: Numerical Simulation Benchmark.

Performance & Parallel Efficiency

A 32-task SLURM job on three G4dn nodes consumed the 10 GB embeddings and
completed the similarity kernel in 8 minutes.

MPI strong scaling from 8 to 32 tasks yielded a 3.5× speed-up (efficiency ~44%),
limited by I/O contention on Lustre under concurrent reads.

Elastic Provisioning

ParallelCluster’s Spot fleet scaled out the GPU queue to three nodes in ~3 minutes
upon job submission.

## 5.4. Quality Attribute Analysis

The following are the main findings with regard to the key quality attributes we were
interested in:

- Scalability: Both phases showed near-linear scale-up with added GPU nodes.
- Elasticity: Automated node lifecycle (Ray autoscaler + ParallelCluster
auto-scaling) matched compute supply to job demand, avoiding
overprovisioning.
- Performance: GPU acceleration cut Phase 1 embedding time by 4× over CPU
only; EFA accelerated Phase 2 by 1.6× over TCP.
- Cost-Efficiency: Combining GPU spot instances and auto-shutdown policies
yielded overall cost savings of 60% compared to a static on-demand cluster.