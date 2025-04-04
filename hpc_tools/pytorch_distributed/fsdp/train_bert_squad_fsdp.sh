#!/bin/bash
#SBATCH -o %x-%J.out
#SBATCH -e %x-%J.error

#SBATCH -J bert_squad_fsdp       # Job name
#SBATCH -o bert_squad_fsdp.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e bert_squad_fsdp.o%j   # Name of stderr output file(%j expands to jobId)

#SBATCH --gres=gpu:a100:2           # Request 2 a100 GPUs; for Tesla --gres=gpu:t4:2
#SBATCH -c 32 #(64 cores per job)
#SBATCH --cpus-per-task=64 
#SBATCH --mem=64G                   # Memory allocation
#SBATCH --time=02:00:00             # Job time limit (2 hours)

echo "SLURM Job launched on $(hostname)"
echo "Loading modules and activating environment"

# Load CUDA and other necessary modules (adjust as needed for your cluster)
module load cuda/11.8 nccl
module load cesga/system 
module load python/3.10.8

#Ensure setup_env.py is run (will create and install if needed)
echo "Running setup_env.py to prepare Python environment"
python3 setup_env.py

#Activate virtual environment
source lightning_env/bin/activate
echo "Environment 'lightning_env' activated"

#Start TensorBoard
tensorboard --logdir=lightning_logs --bind_all &> tensorboard.log &

#Start GPU Monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 10 &> gpu_usage.log &

#Run Training
echo "Starting BERT SQuAD Training Job with FSDP"
torchrun --nnodes=1 --nproc_per_node=2 train_bert_squad_fsdp.py

echo "Training Job Completed"

#Stop GPU Monitoring
pkill -f "nvidia-smi --query-gpu"

#Display GPU Metrics Summary
echo "GPU Usage Summary:"
tail -n 20 gpu_usage.log
