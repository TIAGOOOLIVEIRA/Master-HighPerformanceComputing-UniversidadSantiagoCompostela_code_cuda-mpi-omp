#!/bin/bash
#
#$ -cwd
#$ -j y
#

# submit with something such as sbatch -n1 -c 32 --exclusive --mem=1G -t 9 ./run_parallel_phonebook.sh

for nthreads in 1 2 4 8 16 32; do
	echo --------- using $nthreads threads ---------
	for nreps in {1..3}; do 
		OMP_NUM_THREADS=$nthreads ./parallel_phonebook
	done
done
