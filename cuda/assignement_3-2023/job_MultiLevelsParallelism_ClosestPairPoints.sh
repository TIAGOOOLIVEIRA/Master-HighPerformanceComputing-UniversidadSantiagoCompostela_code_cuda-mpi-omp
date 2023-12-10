 #!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run CUDA applications
# on CESGA's FT-III system.
#----------------------------------------------------
#SBATCH -J gpu_job_MultiLevelsParallelism_ClosestPairPoints       # Job name
#SBATCH -o MultiLevelsParallelism_ClosestPairPoints.o%j   # Name of stdout output file(%j expands to jobId)
#SBATCH -e MultiLevelsParallelism_ClosestPairPoints.o%j   # Name of stderr output file(%j expands to jobId)
#SBATCH -c 32            # Cores per task requested (1 task job)
# Needed 32 cores per A100 demanded
#SBATCH --mem-per-cpu=3G # memory per core demanded
#SBATCH --gres=gpu       # Options for requesting 1GPU
#SBATCH -t 01:30:00      # Run time (hh:mm:ss) - 1.5 hours

# Run the CUDA application
module load cesga/2020 cuda-samples/11.2

./MultiLevelsParallelism_ClosestPairPoints


