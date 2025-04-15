#!/bin/bash
#SBATCH -J bert_squad_ddp             # Job name
#SBATCH -o bert_squad_ddp.o%j         # Standard output
#SBATCH -e bert_squad_ddp.e%j         # Standard error
#SBATCH --gres=gpu:a100:2             # Request 2 A100 GPUs on one node
#SBATCH --cpus-per-task=32            # CPU cores per task
#SBATCH --mem=64G                     # Total memory
#SBATCH --time=02:00:00               # Time limit hrs:min:sec

echo "SLURM Job launched on $(hostname)"

# Load modules
module purge
module load cuda/12.8.0
module load nccl/2.26.2-cuda-12.8.0
module load python/3.10.8

# Activate virtual environment
#source ddp_env/bin/activate
source /mnt/netapp2/Store_uni/home/ulc/cursos/curso370/tools/ddp_env/bin/activate
echo "Environment 'ddp' activated"

#Set dataset/model cache directories to avoid write in home
export TRANSFORMERS_CACHE=/scratch/$SLURM_JOB_ID/transformers
export HF_DATASETS_CACHE=/scratch/$SLURM_JOB_ID/hf_datasets
mkdir -p $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

#Start GPU monitor
LOG_DIR=logs
mkdir -p $LOG_DIR
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 5 > $LOG_DIR/gpu_usage_${SLURM_JOB_ID}.log &
MONITOR_PID=$!

echo "Starting BERT SQuAD DDP Training"
python3 train_bert_squad_ddp.py

#Stop GPU monitor
kill $MONITOR_PID

echo "Training Job Complete"