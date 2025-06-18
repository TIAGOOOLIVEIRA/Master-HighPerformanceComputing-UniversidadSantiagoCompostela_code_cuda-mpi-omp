# Optimizing Vector Algorithm Processing through the integration of Ray IO and HPC in the Cloud



The rapid rise of large-scale AI—whether in natural-language models or computer-vision networks—hinges on turning raw knowledge (text, images, video) into dense numeric arrays (embeddings) that downstream algorithms can process efficiently.  Tokenization and embedding pipelines must handle terabytes—or even petabytes—of data, extract meaningful features, and expose them as vectors for similarity search, graph traversal, or predictive modeling.

Ray has emerged as a compelling orchestration layer for these pipelines: it brings a vibrant community, deep integrations (e.g. Ray Train, Ray Serve, Ray DAG), and a simple autoscaler (e.g. the AWS example-full.yaml) that can spin up EC2 fleets, integrate with AWS ParallelCluster for MPI/Slurm jobs, or launch containerized workloads (via CloudFormation or AWS Batch).  This thesis will explore how Ray can unify the “data-prep” phase—downloading raw assets, sharding and tokenizing them, running encoding/decoding transforms—before handing off subtasks to the most appropriate compute architecture.

On the hardware side, modern CPUs support wide-vector instructions (AVX-512, SVE) that can process dozens of data points per cycle; GPUs (CUDA, cuBLAS, cuVS, CAGRA) deliver thousands of cores for dense and sparse linear algebra; and FPGAs (AWS F1, Xilinx Vitis) can be programmed to accelerate bespoke kernels (e.g. custom similarity metrics, streaming convolution).  Each platform offers a different sweet spot in precision, throughput and cost per operation.  A key contribution of this work is a systematic trade-off analysis—benchmarks that compare AVX on x86, CUDA on NVIDIA GPUs, and FPGA bitstreams for tasks like k-nearest neighbor, graph search, and random-walk simulations.

Beyond raw speed, real-world AI pipelines must glue together multiple stages—data transfer (S3 or FSx-Lustre), pre-processing (Ray tasks), encoding/decoding (vector transforms), model training or fine-tuning, and post-processing/visualization.  We’ll evaluate several end-to-end deployment patterns on AWS: a lightweight Ray cluster for pre-processing, an AWS ParallelCluster for large MPI jobs, an S3 One-Zone bucket for intermediate data, and CloudFormation automations to stitch everything together—even mounting FSx for Lustre or writing Slurm job-submission actors.  Cost models will be derived from actual on-demand vs. spot pricing, and throughput-vs-dollar curves will guide design choices.

Finally, this thesis will present a suite of proof-of-concept benchmarks—on datasets that stretch compute and I/O (e.g. multi-TB image corpora, high-resolution video streams, large-scale knowledge graphs)—to validate similarity search algorithms, diffusion simulations, and graph-based embeddings in heterogeneous environments.  Through this, we aim to demonstrate that a highly interoperable, scalable pipeline—anchored by Ray and leveraging CPU, GPU and FPGA accelerators—can deliver both performance and economic efficiency for next-generation AI workloads.



<img src="../images/AWS_VectorMachine_Ray-HPC-architecture.drawio.svg" alt="AWS Architecture" width="500">
