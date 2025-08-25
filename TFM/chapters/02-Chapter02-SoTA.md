# Chapter 2 - State of the art

The following sections survey how leading‐edge practices in both data
preprocessing and large‐scale simulation converge into a coherent, two‐phase
pipeline that meets current industry standards. First, we examine the Two-Phase
Data-to-HPC Pipeline Framework, highlighting shared preprocessing steps across
domains and the rationale for separating high-throughput ETL (Phase 1) from
MPI/Slurm-driven numeric simulations (Phase 2). Next, we review Cloud-Native
Design Patterns and Tools — from Data Lakes and Ray’s distributed compute
abstractions to AWS ParallelCluster and serverless hybrids — that enable fully
automated, scalable deployments in line with best practices for IaC, CI-CD, and
resilient looped model refinement.

## 2.1. Two-Phase Data-to-HPC Pipeline Framework

Modern scientific and engineering workloads — from molecular dynamics to
genomics — rely on a two-phase data pipeline that first transforms raw,
heterogeneous inputs into clean, structured embeddings (Phase 1) and then drives
compute-intensive simulations or analyses (Phase 2). Table 1 outlines the shared
preprocessing steps — validation, discretization, partitioning, computation, feature
extraction, and serialization — across MD, CFD, FFT, and Genomics.

