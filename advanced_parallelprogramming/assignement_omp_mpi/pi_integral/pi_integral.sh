#!/bin/bash
#SBATCH -J pi_integral
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G


#watch -n 1 squeue -u $USER
#To automate in the interactive session, make sh executable
#chmod +x pi_integral.sh
#module avail cuda

#to run in interactive mode
#./pi_integral.sh


# Load required modules
module purge
module load gcc openmpi/4.0.5_ft3
module load intel vtune
module load intel impi

echo "Compiling with profiling support"
make clean
make

# Run configurations: (MPI_PROCS, OMP_THREADS)
declare -a configs=(
  "2 8"
  "4 4"
  "8 2"
  "16 1"
)

runs=3
integralsize=1000000000

for config in "${configs[@]}"; do
    IFS=' ' read -r mpi_procs omp_threads <<< "$config"
    export OMP_NUM_THREADS=$omp_threads
    echo
    echo "==== Configuration: MPI=$mpi_procs, OMP=$omp_threads ===="
    for ((r=1; r<=runs; r++)); do
        echo "Run $r..."
        mpirun -np $mpi_procs ./pi_integral $integralsize
    done
done

echo "All configurations completed."
