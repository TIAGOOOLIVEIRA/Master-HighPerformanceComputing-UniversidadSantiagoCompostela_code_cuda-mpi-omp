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
    git clone https://github.com/yourusername/pytorch_singleGPU.git
    cd pytorch_singleGPU
    ```

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

To run the example script, use the following command:
```bash
python main.py
```

This script will:
- Initialize a neural network model.
- Load a dataset.
- Train the model on a single GPU.
- Evaluate the model performance.

## File Structure

- `main.py`: The main script to run the example.
- `model.py`: Contains the neural network model definition.
- `data_loader.py`: Handles data loading and preprocessing.
- `train.py`: Contains the training loop and evaluation functions.
- `requirements.txt`: Lists the required Python packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is part of the Master in High Performance Computing at Universidad de Santiago de Compostela.
