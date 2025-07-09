#!/bin/bash
#SBATCH -J stencil
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x stencil.sh

#to run in interactive mode
#./stencil.sh


# Load required modules
module purge
module load gcc openmpi/4.0.5_ft3
module load intel vtune
module load intel impi


mpirun -np stencil 20 1 50

echo "All configurations completed."
