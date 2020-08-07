# SUMMA-algorithm-in-MPI
This program use SUMMA algorithm for multiplying two dense matrices in MPI


Installation:
mpicc summa.c -o summa -lm
mpirun -np 4 ./summa
