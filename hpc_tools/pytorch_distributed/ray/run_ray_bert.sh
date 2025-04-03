#!/bin/bash
#SBATCH --job-name=ray_bert_squad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2           # Request 2 a100 GPUs; for Tesla --gres=gpu:t4:2
#SBATCH --cpus-per-task=64 
#SBATCH --mem=64G                   # Memory allocation
#SBATCH --time=02:00:00             # Job time limit (2 hours)
#SBATCH --time=02:00:00
#SBATCH --output=ray_training.log
#SBATCH --error=ray_training.err

echo "Activating environment and loading modules"
module load cuda/11.8
source activate mypython_env

#Start TensorBoard
tensorboard --logdir=runs --bind_all &> tensorboard.log &

#Start GPU logging
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used --format=csv -l 10 > gpu_usage.log &

#Launch Ray training
echo "Launching Ray application"
python train_bert_squad_ray.py

#Cleanup
pkill -f nvidia-smi
echo "Job completed"
