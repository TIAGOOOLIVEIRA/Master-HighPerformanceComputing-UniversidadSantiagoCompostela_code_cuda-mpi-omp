#!/bin/bash
#
#$ -cwd
#$ -j y
#

# submit with something such as sbatch -n1 -c4 --mem=1G -t 9 ./run_prod_cons.sh

echo ----------------------------------
echo OMP_NUM_THREADS=2 ./prod_cons_atom
for nreps in {1..3}; do OMP_NUM_THREADS=2 ./prod_cons_atom; done
echo ----------------------------------
echo OMP_NUM_THREADS=2 ./prod_cons_cond
for nreps in {1..3}; do OMP_NUM_THREADS=2 ./prod_cons_cond; done

echo ----------------------------------
echo OMP_NUM_THREADS=2 ./prod_cons_atom -f
for nreps in {1..3}; do OMP_NUM_THREADS=2 ./prod_cons_atom -f; done
echo ----------------------------------
echo OMP_NUM_THREADS=2 ./prod_cons_cond -f
for nreps in {1..3}; do OMP_NUM_THREADS=2 ./prod_cons_cond -f; done

echo ----------------------------------
echo OMP_NUM_THREADS=8 ./prod_cons_atom -f
for nreps in {1..3}; do OMP_NUM_THREADS=8 ./prod_cons_atom -f; done
echo ----------------------------------
echo OMP_NUM_THREADS=8 ./prod_cons_cond -f
for nreps in {1..3}; do OMP_NUM_THREADS=8 ./prod_cons_cond -f; done
