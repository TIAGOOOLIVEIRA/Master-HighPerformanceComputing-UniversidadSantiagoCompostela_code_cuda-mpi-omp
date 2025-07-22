**AWS Data Enablers for Synergy between Ray and HPC:**

AWS provides a robust ecosystem that supports the synergy between Ray clusters (often used for data preprocessing and AI/ML purposes) and traditional HPC workloads:

- Infrastructure as Code (IaC) and Automation:
- AWS ParallelCluster: A key service for setting up and managing HPC clusters on AWS, enabling users to define and provision the necessary infrastructure (compute, networking, storage) using IaC principles [@aws_compute_starccm]
- It can integrate with services like Amazon FSx for Lustre and EFA
- AWS CloudFormation: Used to provision and manage AWS resources, supporting automated and reproducible infrastructure deployments for both HPC and Ray clusters
- AWS Batch & Amazon EKS: For managing containerized workloads. Ray clusters can be deployed on Amazon EKS [@nguyen2023building]. AWS Batch helps run HPC and big data applications without managing infrastructure

**Interoperability and Data Flow:**

- Amazon S3: Serves as a centralized, highly scalable, and cost-effective data lake for raw and processed data, accessible by all AWS services, including Ray and HPC clusters
- This alleviates the need for extensive data transfers between different compute environments.
- AWS DataSync: Facilitates secure and efficient transfer of large datasets from on-premises environments to Amazon S3, bridging the gap between existing data centers and cloud HPC/ML infrastructure
- Amazon FSx for Lustre & Amazon EFS: Provide high-performance, shared file systems necessary for HPC workloads and for efficient data access by Ray clusters [@khanuja2022applied]
- Amazon SageMaker HyperPod: A managed service that enables scalable and resilient distributed AI/ML workloads using Ray jobs [@vinciguerra2025rayhyperpod]
- This demonstrates AWS's direct support for integrating Ray with its managed ML services, offering enhanced resiliency and auto-resume capabilities crucial for long-running tasks
- Hybrid Execution Models: AWS supports hybrid architectures like Mashup, which leverages both traditional VM-based (EC2) and serverless (Lambda) platforms to execute scientific workflows
- This approach can optimize both execution time (average 34% reduction) and cost (average 43% reduction) by intelligently placing tasks on the most suitable platform (e.g., bursty, stateless tasks to serverless; long-running, stateful tasks to VMs). This concept applies directly to integrating Ray (for AI/ML preprocessing, training) with HPC (for traditional simulations) by selecting the optimal compute environment for each stage of a complex workflow.

**Objective**
By sticking to the architectural design of two phased pipeline, where data crunching relies on the phase 1 and numerical simulation in the phase 2 [@roy2017massively], interoperability between these two application context-domains becomes the key attribute to ensure data flows efficiently from one side to another.

This investigation aims to tackle not only processing aspects for both phases, but also to address the integration needed for this proposed architecture leveraging the proper service tool elements in the AWS cloud platform.

<img src="../images/Anyscale-Ray-Gen-AI-6.png" alt="HPC AWS" width="500">
<img src="../images/2024-aws-pcs-1-diagram.png" alt="HPC AWS" width="500">


AWS Data Enablers for Interoperability and Automation: Amazon Web Services (AWS) provides a comprehensive set of services to enable HPC, AI, and ML workloads in the cloud, offering virtually unlimited infrastructure and fast networking [@khanuja2022applied]


**Infrastructure as Code (IaC) and Automation:**

- AWS ParallelCluster allows users to set up HPC clusters using CLI or CloudFormation APIs, supporting automated resource provisioning and management
- AWS Batch helps run HPC and Big Data applications without the need to manage underlying infrastructure, using containerized applications and integrating with workflow services like AWS Step Functions
- Ephemeral resources and managed services (e.g., AWS Glue for Spark or Amazon SageMaker APIs for ML training/inference) reduce the need for manual infrastructure provisioning and management
- Best practices advocate for automation through Continuous Integration/Continuous Delivery (CI/CD) and ensuring work reproducibility by managing inputs, environment configurations, and packages

**Data Management and Transfer:**

- Amazon S3 is a recommended, cost-effective storage for large genomics datasets, as it handles file-based data efficiently
- AWS DataSync facilitates secure and scalable transfer of petabyte-scale data to Amazon S3 from on-premises
- Amazon FSx for Lustre provides a fully managed, high-performance file system optimized for HPC workloads, with seamless and fast access to data from Amazon S3 or on-premises, including POSIX support and backup features


**Interoperability between Ray and HPC Clusters on AWS:**

- Amazon SageMaker HyperPod offers a powerful solution for running Ray jobs for scalable and resilient distributed AI workloads, combining Ray's flexibility with HyperPod's robust infrastructure, enhanced resiliency, and auto-resume capabilities for long-running, resource-intensive tasks
- AWS Glue for Ray allows processing large datasets with Python within the AWS Glue environment
- AWS provides various compute instances (EC2, especially GPU-based P-family instances like p4d with Ampere A100 GPUs) that can be leveraged for both traditional HPC and ML workloads
- Networking services like Elastic Fabric Adapter (EFA), used with cluster placement groups, minimize latency between nodes, which is crucial for tightly coupled HPC and distributed AI applications
- The Mashup strategy exemplifies interoperability by leveraging a hybrid execution model combining serverless (AWS Lambda) and VM-based (Amazon EC2) platforms for HPC workflows, significantly reducing execution time (average 34%) and cost (average 43%) compared to traditional cluster execution [@koons2021mashup]
- This approach allows for dynamic platform selection based on task suitability
- Tools like Amazon Genomics CLI provide purpose-built open-source solutions for processing raw genomics data at petabyte scale directly in the cloud, streamlining complex workflows
- For remote access and visualization, NICE DCV enables remote desktop sessions to HPC clusters, providing a graphical interface for researchers
- These AWS services and architectural patterns enable a flexible and scalable environment where different computational paradigms (traditional HPC, Big Data, and AI/ML with frameworks like Ray) can interoperate to solve complex scientific challenges.
