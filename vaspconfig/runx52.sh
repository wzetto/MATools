#!/bin/sh
#$ -cwd
#$ -V -S /bin/bash
#$ -pe smp 52
#$ -q x52
#$ -N jobname
mpirun -np 52 vasp

