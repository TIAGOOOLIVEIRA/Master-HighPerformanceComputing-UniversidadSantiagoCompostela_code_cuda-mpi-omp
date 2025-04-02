#!/bin/bash
#SBATCH --job-name=bert_squad_fsdp
#SBATCH --gres=gpu:a100:4           # Request 2 a100 GPUs; for Tesla --gres=gpu:t4:2
#SBATCH -c 32                       # 32 CPU cores
#SBATCH --mem=64G                   # Memory allocation
#SBATCH --time=02:00:00             # Job time limit (2 hours)
#SBATCH --output=training_output_fsdp.log
#SBATCH --error=training_error_fsdp.log

echo "SLURM Job launched on $(hostname)"
echo "Loading modules and activating environment"

# Load CUDA and other necessary modules (adjust as needed for your cluster)
module load cuda/11.8 nccl

#Ensure setup_env.py is run (will create and install if needed)
echo "Running setup_env.py to prepare Python environment"
python setup_env.py

#Activate virtual environment
source lightning_env/bin/activate
echo "Environment 'lightning_env' activated"

#Start TensorBoard
tensorboard --logdir=lightning_logs --bind_all &> tensorboard.log &

#Start GPU Monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 10 &> gpu_usage.log &

#Run Training
echo "Starting BERT SQuAD Training Job with FSDP"
torchrun --nnodes=1 --nproc_per_node=4 train_bert_squad_fsdp.py

echo "Training Job Completed"

#Stop GPU Monitoring
pkill -f "nvidia-smi"

#Display GPU Metrics Summary
echo "GPU Usage Summary:"
tail -n 20 gpu_usage.log
