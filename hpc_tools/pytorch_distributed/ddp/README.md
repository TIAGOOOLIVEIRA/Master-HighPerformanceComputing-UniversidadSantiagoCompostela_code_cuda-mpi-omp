# PyTorch Distributed training - leveraging DDP

This project documents a task and experiments focused on migrating a PyTorch Lightning-based multi-GPU training pipeline to a native **PyTorch DDP** implementation, executed on a **Slurm-managed HPC cluster**.

## Prerequisites

- Python ≥ 3.9
- Linux-based OS
- CUDA-compatible GPU (e.g., NVIDIA A100 or V100)
- Access to an HPC cluster with SLURM


## Purpose

- Achieve fine-grained control over GPU allocation and parallelism.
- Overcome PyTorch Lightning’s rigidity in cluster orchestration.
- Integrate TensorBoard and Slurm into a flexible distributed workflow.

The experiment was successful, and the final distributed training job completed in:

> ✅ **Total training time: 2896.58 seconds**

## Installation

1. Clone the repository:
```bash
    git clone https://github.com/yourusername/Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp.git
    cd Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp/hpc_tools/pytorch_distributed/ddp

```


2. For manual installation - Create a virtual environment and activate it:
```bash
    python3 -m venv ddp_env
    source ddp_env/bin/activate

```

3. Install the required packages:
```bash
    
    Install required packages using:
    pip3 install -r requirements.txt
    pip3 install -r requirements_tensorboard.txt
```

## Usage
Before running the example script, ensure that your environment is properly set up. Follow these steps:

1. For automated instalation - Set up the environment by running the `setup_env.py` script:
```bash
    
    Interactive Mode
    chmod +x train_bert_squad_ddp.sh
    ./train_bert_squad_ddp.sh
```

2. Submit the training job using the SLURM script:
```bash
    sbatch train_bert_squad_ddp.sh
    watch -n 1 squeue
```

This will:
- Prepare the environment by installing necessary dependencies.
- Submit the job to train a BERT model on the SQuAD dataset using two GPU cards in a single node.

Make sure to check the SLURM output files for job status and results.


This script will:
- Initialize a neural network model.
- Load a dataset.
- Train the model on a single GPU.
- Evaluate the model performance.
- Generate logs of execution and sabe in the folder logs.
- Save the model in the folder bert_squad_ddp_trained.
- Save the model dataset in the folder cached_squad.

## File Structure

- `requirements_tensorboard.txt`: python libraries for tensorboard.
- `requirements.txt`: general libraries instaltion.
- `train_bert_squad_ddp.py`: python script for the DDP pytorch trainig.
- `train_bert_squad_ddp.sh`: shell script for the slurm job submition so the python script is managed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Logs

- tail -n 100 logs/gpu_usage_11807744.log
- tail -n 100 tensorboard_ddp_bert.log
- tail -n 100 bert_squad_ddp.o11807744

- $ ls -la bert_squad_ddp_trained
```bash
(ddp_env) [curso370@login209-18 tools]$ ls -la bert_squad_ddp_trained
total 427368
drwxr-xr-x 2 curso370 ulc      4096 Apr 15 01:55 .
drwxr-xr-x 7 curso370 ulc      4096 Apr 15 01:55 ..
-rw-r--r-- 1 curso370 ulc       685 Apr 15 01:55 config.json
-rw-r--r-- 1 curso370 ulc 435644909 Apr 15 01:55 pytorch_model.bin
-rw-r--r-- 1 curso370 ulc       125 Apr 15 01:55 special_tokens_map.json
-rw-r--r-- 1 curso370 ulc       366 Apr 15 01:55 tokenizer_config.json
-rw-r--r-- 1 curso370 ulc    231508 Apr 15 01:55 vocab.txt
(ddp_env) [curso370@login209-18 tools]$ 

(ddp_env) [curso370@login209-18 tools]$ ls -la logs
total 284
drwxr-xr-x 2 curso370 ulc  4096 Apr 15 01:06 .
drwxr-xr-x 7 curso370 ulc  4096 Apr 15 01:55 ..
-rw-r--r-- 1 curso370 ulc 32298 Apr 14 13:10 gpu_usage_11794280.log
-rw-r--r-- 1 curso370 ulc 67766 Apr 14 14:36 gpu_usage_11799538.log
-rw-r--r-- 1 curso370 ulc  4042 Apr 14 15:25 gpu_usage_11803197.log
-rw-r--r-- 1 curso370 ulc 69184 Apr 14 19:13 gpu_usage_11804048.log
-rw-r--r-- 1 curso370 ulc 89693 Apr 15 01:55 gpu_usage_11807744.log
(ddp_env) [curso370@login209-18 tools]$ 
```

## Acknowledgements

This project is part of the Master in High Performance Computing at Universidad de Santiago de Compostela.

## References

- [**Deep Learning with PyTorch** – Eli Stevens, Luca Antiga, Thomas Viehmann (O’Reilly)](https://www.oreilly.com/library/view/deep-learning-with/9781492045519/)

- [**Mastering PyTorch** – Ashish Ranjan Jha (O’Reilly)](https://www.oreilly.com/library/view/mastering-pytorch/9781789138225/)

- [**Ray Train Documentation** – Official Ray Docs](https://docs.ray.io/en/latest/train/train.html)

- [**SLURM sbatch Command Reference** – SchedMD](https://slurm.schedmd.com/sbatch.html)
