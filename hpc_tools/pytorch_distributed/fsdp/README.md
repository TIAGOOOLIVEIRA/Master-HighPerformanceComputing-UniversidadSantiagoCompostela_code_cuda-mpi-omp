# PyTorch Distributed GPU training with Lightning (FSDP)



## Prerequisites

- Python 3.x
- PyTorch, Lightning
- CUDA Toolkit (for GPU support)
- NVIDIA GPU with CUDA support

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp.git
    cd Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp/hpc_tools/pytorch_distributed/fsdp


2. For manual installation - Create a virtual environment and activate it:
    ```bash
    python -m venv lightning_env
    source lightning_env/bin/activate  # On Windows use `lightning_env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Before running the example script, ensure that your environment is properly set up. Follow these steps:

1. For automated instalation - Set up the environment by running the `setup_env.py` script:
    ```bash
    python setup_env.py
    ```

2. Submit the training job using the SLURM script:
    ```bash
    sbatch train_bert_squad_fsdp.sh
    ```

This will:
- Prepare the environment by installing necessary dependencies.
- Submit the job to train a BERT model on the SQuAD dataset using lightning FSDP strategy on the GPU cards made available in the slurm job argument --gres.

Make sure to check the SLURM output files for job status and results.


This script will:
- Initialize a neural network model.
- Load a dataset.
- Train the model on the GPU cards made available.
- Save the model.

## File Structure

- `train_bert_squad.py`: Script to train a BERT model on the SQuAD dataset using a single GPU.
- `train_bert_squad_fsdp.sh`: Script to submit the job of train a BERT model on the SQuAD dataset using multiple GPU cards.
- `setup_env.py`: Sets up the environment by installing necessary dependencies and configurations.
- `requirements.txt`: Lists the required Python packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Logs

- tail -n 100 gpu_usage.log
- tail -n 100 tensorboard.log
- tail -n 100 training_error_fsdp.log
- tail -n 100 training_output_fsdp.log

- $ ls -la bert_squad_trained
```...
```

## Acknowledgements

This project is part of the Master in High Performance Computing at Universidad de Santiago de Compostela.
