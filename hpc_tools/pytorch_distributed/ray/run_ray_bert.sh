#!/bin/bash
#SBATCH -J ray_bert_squad       # Job name
#SBATCH -o ray_bert_squad.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e ray_bert_squad.o%j   # Name of stderr output file(%j expands to jobId)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2           # Request 2 a100 GPUs; for Tesla --gres=gpu:t4:2
#SBATCH --cpus-per-task=64 
#SBATCH --mem=64G                   # Memory allocation
#SBATCH --time=02:00:00             # Job time limit (2 hours)
#SBATCH --time=02:00:00

echo "Activating environment and loading modules"
module load cuda/11.8 nccl
module load cesga/system 
module load cesga python/3.10.8

#Ensure setup_env.py is run (will create and install if needed)
#echo "Running setup_env_ray.py to prepare Python environment"
#python3 setup_env_ray.py

#Activate virtual environment
source ray_env/bin/activate
echo "Environment 'ray_env' activated"
tensorboard --logdir=runs --bind_all &> ray_tensorboard.log &

#Start GPU logging
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used --format=csv -l 10 > ray_gpu_usage.log &

#Launch Ray training
echo "Launching Ray application"
python3 train_bert_squad_ray.py

#Cleanup
pkill -f nvidia-smi
echo "Job completed"
