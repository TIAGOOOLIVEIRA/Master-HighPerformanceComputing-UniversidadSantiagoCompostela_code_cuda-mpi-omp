#!/bin/bash
#
#$ -cwd
#$ -j y
#

# submit with something such as sbatch -n1 -c 32 --exclusive --mem=1G -t 9 ./run_stack.sh -w 800

for nthreads in 1 2 4 8 16 32; do
	echo --------- mutex_concurrent_stack with $nthreads threads ---------
	for nreps in {1..3}; do 
		OMP_NUM_THREADS=$nthreads ./mutex_concurrent_stack $*
	done
	echo --------- atomic_concurrent_stack with $nthreads threads ---------
	for nreps in {1..3}; do 
		OMP_NUM_THREADS=$nthreads ./atomic_concurrent_stack $*
	done
	echo --------- atomic_flag_concurrent_stack with $nthreads threads ---------
	for nreps in {1..3}; do 
		OMP_NUM_THREADS=$nthreads ./atomic_flag_concurrent_stack $*
	done
	echo ==================================================
done
