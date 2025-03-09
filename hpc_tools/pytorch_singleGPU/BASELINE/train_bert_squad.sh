#!/bin/bash
#SBATCH --job-name=bert_squad_training
#SBATCH --gres=gpu:a100           # Request A100 GPU
#SBATCH -c 32                     # Request 32 CPU cores
#SBATCH --mem=64G                 # Memory allocation
#SBATCH --time=00:59:00           # Job time limit (59 min)
#SBATCH --output=training_output.log
#SBATCH --error=training_error.log

echo "Loading Env Modules"
module load cuda/11.8 nccl

#Setup Environment
if [ ! -d "mypython_env" ]; then
    echo "Setting up Python environment..."
    python setup_env.py
else
    echo "Virtual environment found!"
fi

#Activate Env
source mypython_env/bin/activate

#Start TensorBoard
tensorboard --logdir=runs/bert_squad_single_gpu --bind_all &> tensorboard.log &

#GPU Monitoring
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 10 &> gpu_usage.log &

#Run Training
echo "Starting BERT SQuAD Training Job"
python train_bert_squad.py

echo "Training Job Completed"

#Stop GPU Monitoring
pkill -f "nvidia-smi"

#Display GPU Metrics Summary
echo "GPU Usage Summary:"
tail -n 25 gpu_usage.log
