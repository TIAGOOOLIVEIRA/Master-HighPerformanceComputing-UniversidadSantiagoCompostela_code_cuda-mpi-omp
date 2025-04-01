#!/bin/bash
#SBATCH --job-name=bert_squad_fsdp
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH -c 32                     # 32 CPU cores
#SBATCH --mem=64G                # Memory allocation
#SBATCH --time=02:00:00           # Job time limit (2 hours)
#SBATCH --output=training_output_fsdp.log
#SBATCH --error=training_error_fsdp.log

echo "Loading Environment Modules"
module load cuda/11.8 nccl
source activate mypython_env

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
