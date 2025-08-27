# Conclusions and Future Work


This thesis set out to demonstrate that a two‐phase, cloud-native architecture —
using Ray for high-throughput data preparation and AWS ParallelCluster for
compute-intensive numeric simulation — can deliver significant gains in scalability,
elasticity, performance, and cost-efficiency. Through our experiments, we showed
that GPU-accelerated Ray clusters scale nearly linearly when extracting
high-dimensional DNA embeddings, while Slurm/MPI jobs on EFA-enabled HPC
instances achieve strong parallel efficiency for downstream similarity kernels.
Automated node lifecycle management kept resource utilization aligned to demand,
and leveraging AWS spot instances cut infrastructure costs by over 60%. These
results validate our approach of decoupling “last-mile” feature engineering from raw
simulation workloads, fulfilling the objective of “Optimizing Vector Algorithm
Processing through the Integration of Ray IO and HPC in the Cloud.”


Future Work

Building on these findings, we identify three promising avenues for further
investigation:

### 6.1. Evaluating Serverless ETL (AWS Glue) vs. Ray

While Ray Data proved ideal for GPU-accelerated, Python-native preprocessing,
fully managed PaaS offerings like AWS Glue now support Spark-under-the-hood
jobs and offer emerging GPU acceleration options. A systematic comparison should
explore:

- GPU Support: Can Glue’s emerging GPU-enabled workers match Ray’s
actor-based GPU throughput?
- Language Interoperability: How easily can Glue integrate C++ UDFs for
custom kernels versus Ray’s Python-first model?
- Development Experience: Is Glue’s auto-scaling, schema inference, and job
orchestration simpler for end-users than maintaining a Ray autoscaler cluster?

Answering these questions will clarify which PaaS best balances general-purpose
data transformation and deep AI embedding tasks.

### 6.2. Conversational Agents for End-to-End IaC

Reducing manual DevOps friction is the next frontier. By integrating a domain-aware
LLM agent into the pipeline, users could simply describe their need — “Prepare DNA embeddings for sample X and run similarity simulation” — and automatically
generate or update the required CloudFormation/CDK templates and job submission
scripts for both Phase 1 and Phase 2. Key research questions include:

- Prompt-to-IaC Reliability: How accurately can an LLM translate
natural-language requirements into valid, secure IaC?
- Verification & Guardrails: What static analysis or policy checks are needed to
ensure the generated templates meet compliance and cost-control standards?
- Operational Feedback: Can the agent monitor workloads, interpret failures, and
iteratively refine both code and infrastructure definitions?

### 6.3. Advanced Heterogeneous Programming

Our current pipeline leverages GPUs extensively, but the growing maturity of FPGA
and Apple Silicon accelerators offers additional performance and cost advantages:

- FPGA Workloads: By offloading hot kernels — e.g., k-nearest neighbor search
or sparse graph traversals — to AWS F1 instances programmed via Xilinx Vitis
or SYCL, we may achieve orders-of-magnitude speedups for specific
embedding and simulation steps.
- Apple M-Series SoCs: The unified memory architecture and Apple’s AMX and
Neural Engine could transform local development and edge prototyping of our two-phase pipeline. Exploring Ray’s affinity scheduling on macOS runners or
Apple-optimized libraries (e.g., mlcompute) may open new frontiers for
desktop-scale HPC.

By pursuing these directions — benchmarking managed ETL platforms, embedding
conversational DevOps, and embracing heterogeneous hardware — we can extend
our two-phase framework into an even more adaptable, automated, and
high-performance solution for the next generation of data-intensive scientific and AI
workflows.