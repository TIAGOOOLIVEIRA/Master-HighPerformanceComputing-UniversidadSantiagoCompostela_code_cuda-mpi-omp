# 3. Bridging Ray and HPC on AWS

## AWS Data Enablers for Ray–HPC Integration

To make our two-phase pipeline work seamlessly, AWS offers an ecosystem of services that glue Ray’s preprocessing power to traditional HPC simulations. At its core is **Amazon S3**—a centralized data lake where raw reads and embeddings live—and **AWS Lake Formation**, which governs access and metadata. From there, Ray tasks can load, clean, and vectorize data, then hand off compact embeddings to HPC nodes for numerical simulation.

## Infrastructure as Code and Automation

We automate everything with IaC. **AWS ParallelCluster** lets us define our HPC cluster—compute, EFA networking, FSx for Lustre storage—in a single template \[@aws\_compute\_starccm]. **CloudFormation** does the same for Ray clusters, AWS Glue jobs, and SageMaker endpoints. For containerized workloads, **AWS Batch** and **Amazon EKS** simplify deployment: Ray clusters can run on EKS \[@nguyen2023building], while Batch lets us submit both big-data pipelines and MPI/Slurm jobs without managing servers.

## Interoperability and Data Flow

Data flows wherever it’s needed. We stage terabytes of reads into S3, and use **AWS DataSync** to pull on-premises archives into the cloud securely. **FSx for Lustre** and **Amazon EFS** then mount that data for both Ray workers and HPC nodes \[@khanuja2022applied]. When Ray finishes embedding generation, jobs can push vectors back to S3 or directly into FSx, and the HPC cluster reads them in for spectral solvers or graph algorithms. For long-running AI tasks, **SageMaker HyperPod** runs Ray jobs with built-in resiliency and auto-resume \[@vinciguerra2025rayhyperpod].

## Hybrid Execution Models

Sometimes the best performance comes from mixing paradigms. The **Mashup** approach, for example, splits tasks between EC2 VMs and Lambda functions—burst-y jobs go serverless, while stateful jobs land on EC2—cutting runtime by 34% and cost by 43% \[@koons2021mashup]. We apply the same tactic: Ray handles the elastic, stateless preprocessing, and HPC handles the heavy, stateful simulations.

## Objective and Scope

By sticking to our two-phase architecture—Ray for data prep and AWS-HPC for numeric simulation \[@roy2017massively]—we ensure that each service plays to its strengths. This investigation will not only benchmark performance across both phases but also demonstrate how AWS’s data enablers and hybrid models deliver a truly interoperable, scalable pipeline for next-generation DNA sequencing simulations.





<img src="../images/Anyscale-Ray-Gen-AI-6.png" alt="HPC AWS" width="500">
<img src="../images/2024-aws-pcs-1-diagram.png" alt="HPC AWS" width="500">
