#!/bin/bash
#SBATCH -J mxvnm
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x mxvnm.sh

#to run in interactive mode
#./mxvnm.sh


# Load required modules
module purge
module load gcc openmpi/4.0.5_ft3
module load intel vtune
module load intel impi

# (MPI_PROCS, OMP_THREADS)
declare -a configs=(
  "1 1"
  "2 8"
  "4 4"
  "8 2"
  "16 1"
)

# (N M) values: N ≠ M in all combinations
declare -a sizes=(
  "1000 10000"
  "10000 1000"
  "1000 100000"
)

# Run each configuration multiple times
runs=3

for config in "${configs[@]}"; do
    IFS=' ' read -r mpi_procs omp_threads <<< "$config"
    export OMP_NUM_THREADS=$omp_threads
    echo
    echo "==== Configuration: MPI=$mpi_procs, OMP=$omp_threads ===="
    for size in "${sizes[@]}"; do
        IFS=' ' read -r N M <<< "$size"
        echo "-- Matrix size N=$N, M=$M"
        for ((r=1; r<=runs; r++)); do
            echo "Run $r..."
            mpirun -np $mpi_procs ./mxvnm $N $M
        done
    done
done

echo "All configurations completed."
