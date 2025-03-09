# PyTorch Single GPU Example

This repository contains an example of how to use PyTorch with a single GPU for high-performance computing tasks. The code demonstrates basic operations and training of a neural network on a single GPU.

## Prerequisites

- Python 3.x
- PyTorch
- CUDA Toolkit (for GPU support)
- NVIDIA GPU with CUDA support

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp.git
    cd Master-HighPerformanceComputing-UniversidadSantiagoCompostela_code_cuda-mpi-omp/hpc_tools/pytorch_singleGPU


2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Before running the example script, ensure that your environment is properly set up. Follow these steps:

1. Set up the environment by running the `setup_env.py` script:
    ```bash
    python setup_env.py
    ```

2. Submit the training job using the SLURM script:
    ```bash
    sbatch train_bert_squad.sh
    ```

This will:
- Prepare the environment by installing necessary dependencies.
- Submit the job to train a BERT model on the SQuAD dataset using a single GPU.

Make sure to check the SLURM output files for job status and results.


This script will:
- Initialize a neural network model.
- Load a dataset.
- Train the model on a single GPU.
- Evaluate the model performance.

## File Structure

- `train_bert_squad.py`: Script to train a BERT model on the SQuAD dataset using a single GPU.
- `train_bert_squad.sh`: Script to submit the job of train a BERT model on the SQuAD dataset using a single GPU.
- `setup_env.py`: Sets up the environment by installing necessary dependencies and configurations.
- `requirements.txt`: Lists the required Python packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is part of the Master in High Performance Computing at Universidad de Santiago de Compostela.
