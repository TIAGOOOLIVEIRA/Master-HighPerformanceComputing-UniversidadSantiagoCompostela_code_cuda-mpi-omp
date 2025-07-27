Notes and highlights from papers to reference in the thesis text

fernandezfraga2024applying - Applying Dynamic Balancing to Improve the Performance of MPI Parallel Genomics Applications


- Load imbalance is a common issue in parallel genomics applications due to heterogeneous processing costs and variable biological sequence sizes [fernandezfraga2024applying]


**Distributed Computing and Programming Models**:

- Ray is an open-source, unified compute framework that scales AI and Python workloads from laptops to cloud clusters with minimal code changes
- It simplifies distributed computing challenges such as hardware failures, managing large cluster dependencies, job scheduling, and GPU optimization, particularly for training large language models. Ray provides core abstractions like tasks, actors, and objects, and high-level libraries like Ray AIR (AI Runtime) to unify ML workflows from data processing to model serving
- Hadoop and Spark frameworks, popular for Big Data analysis, can be efficiently run on HPC platforms in the cloud to avoid large data transfers and leverage powerful resources
- OpenMP and MPI are standard parallel programming models that work together (MPI+OpenMP tasking) for high performance and scalability on modern heterogeneous systems
- Hybrid Execution Models: New strategies like Mashup leverage both traditional VM-based cloud computing platforms (e.g., AWS EC2) and serverless platforms (e.g., AWS Lambda) to execute scientific workflows in a hybrid fashion
- This approach has shown an average of 34% reduction in execution time and 43% reduction in cost for widely-used HPC workflows by mitigating serverless-specific challenges like stateless execution, execution timeouts, and cold-start overheads
- Data Movement and I/O: Unconventional architectures like processing-in-memory are being explored to overcome data movement restrictions inherent in load-store architectures (CPU, GPU, FPGA)
- Cloud storage services like Amazon S3, EFS, and FSx offer scalable, high-throughput, and low-latency options to manage large datasets and reduce I/O overheads




**DNA Sequencing computation challenges:**

Processing whole‐genome data involves ingesting terabytes of error-prone short reads and running multi-stage pipelines—alignment, cleaning, sorting, and variant calling—across 15–20 tools that were not built for scale. 

Cloud HPC supplies virtually unlimited compute and storage, crushing through these bottlenecks: massive parallel alignment (e.g., GPU-accelerated BWA or DeepVariant), data cleaning at petabyte scale, and high-throughput variant inference. 

AWS offers C-, M-, and R-family instances for CPU-bound steps, G-family for GPU-enabled base-calling and deep-learning callers, and P4d instances—thousands of A100 GPUs with Petabit networking — for training and inference of neural variant detectors with near-linear scaling. 

The Elastic Fabric Adapter (EFA) further accelerates MPI-style workloads (e.g., distributed genome assembly), slashing end-to-end runtimes and moving us closer to clinical 1–2-day turnaround.




**AWS ParallelCluster for a CFD application**

This reference architecture uses AWS ParallelCluster to deploy a turnkey HPC environment for running Siemens’ Simcenter STAR-CCM+ CFD application. It automates provisioning of C5n instances with EFA for low-latency MPI, mounts an Amazon FSx for Lustre parallel file system for high-throughput I/O, and leverages 100 Gbps networking—all in under 15 minutes. Users then install STAR-CCM+ on the cluster and submit simulation jobs, with optional NICE DCV desktops for remote visualization. The strategy emphasizes rapid, repeatable deployment of optimized HPC resources to accelerate large-scale CFD workloads in the cloud [@aws_compute_starccm].

<img src="../images/AWS_HPC_ParallelCluster_StarccmFig1.png" alt="HPC AWS" width="500">

<img src="../images/hpcblog-53-fig4.png" alt="HPC client - HPC" width="500">



### Synthetic Data Generation for CFD
Ray’s modularity shines in data augmentation:

    Deep Generative Models (StyleGAN, PDE-aware VAEs) run as Ray actors to produce synthetic flow fields or mesh deformations.

    On-the-fly Streaming: Generated images or volume grids are chunked and fed directly into CFD solvers (OpenFOAM, ANSYS) via Ray tasks, enabling rapid scenario exploration without manual data staging.


CFD image preprocessing with DeepLearning/Encoding - geometric (Stereolithography - STL - 3D points))
    eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures https://arxiv.org/abs/2506.09994
    https://cfd-on-pcluster.workshop.aws
    https://insidehpc.com/2017/05/hpc-service-high-performance-video-rendering/
    Learning three-dimensional flow for interactive aerodynamic design dl.acm.org/doi/10.1145/3197517.3201325
    https://www.researchgate.net/figure/mage-from-an-Autodesk-tutorial-displaying-flow-lines-over-a-car_fig8_331929090
        2 ACTIVE MEMS-BASED FLOW CONTROL




Genomics for encoding GNA sequencing: LLM or Graph Neuron Netork (tasks of DNA classification, interpretation of structural, )
    pattern mathing over GNN - after encodding (graph vector embeddings)
    visualizie results: NICE DVC to load mesh (from OpenFOAM) in ParaView 
    Genomic benchmarks: a collection of datasets for genomic sequence classification pmc.ncbi.nlm.nih.gov/articles/PMC10150520/

    160-fold acceleration of the Smith-Waterman algorithm using a field programmable gate array (FPGA) pmc.ncbi.nlm.nih.gov/articles/PMC1896180/pdf/1471-2105-8-185.pdf
    https://www.oreilly.com/library/view/basic-applied-bioinformatics/9781119244332/c10.xhtml
    Genomics, High Performance Computing and Machine Learning www.researchgate.net/publication/352869810_Genomics_High_Performance_Computing_and_Machine_Learning
    computationonal exome and genome analysis api.pageplace.de/preview/DT0400.9781498775991_A30884522/preview-9781498775991_A30884522.pdf
        https://www.researchgate.net/publication/320959019_Computational_Exome_and_Genome_Analysis
    https://www.researchgate.net/publication/230846804_Parallel_Iterative_Algorithms_From_Sequential_to_Grid_Computing
        Parallel Iterative Algorithms: From Sequential to Grid Computing books.google.com.pa/books?id=ft7E5hiIzDAC&printsec=frontcover#v=onepage&q=protein&f=false
    basic applied bioinformatics

    https://www.internationalgenome.org/
        https://registry.opendata.aws/1000-genomes/

    References for Dataset 
    registry.opendata.aws
    protein 
        https://github.com/PacktPublishing/Applied-Machine-Learning-and-High-Performance-Computing-on-AWS/blob/main/Chapter12/protein-secondary-structure-model-parallel.ipynb
        Genomic benchmarks: a collection of datasets for genomic sequence classification
        https://github.com/rieseberglab/fastq-examples
        https://learn.gencore.bio.nyu.edu/ngs-file-formats/fastq-format/


https://www.genome.gov/about-genomics/fact-sheets/Genomic-Data-Science
https://registry.opendata.aws/
https://registry.opendata.aws/1000-genomes/