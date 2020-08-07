/******************************************************************************************
 *
 *	Filename:	summa.c
 *	Purpose:	A paritally implemented program for MSCS6060 HW.
 *Students will complete the program by adding SUMMA implementation for matrix
 *multiplication C = A * B. Assumptions:    A, B, and C are square matrices n by
 *n; the total number of processors (np) is a square number (q^2). To compile,
 *use mpicc -o summa summa.c To run, use mpiexec -n $(NPROCS) ./summa
 *********************************************************************************************/

#include "mpi.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define min(a, b) ((a < b) ? a : b)
// SZ (4200, 5040) need to divisible by sqrt(num of processor)  && num of processor can be
// sqrt.
#define SZ 4200
// Each matrix of entire A, B, and C is SZ by SZ. Set a small value for
// testing, and set a large value for collecting experimental data.

/**
 *   Allocate space for a two-dimensional array
 */

void printmatrix(double **matrix, int block_sz)
{
  int i, j;
  for (i = 0; i < block_sz; i++)
  {
    for (j = 0; j < block_sz; j++)
      printf("%.2lf\t", matrix[i][j]);
    printf("\n");
  }
  printf("\n");
}
double **alloc_2d_double(int n_rows, int n_cols)
{
  int i;
  double **array;
  array = (double **)malloc(n_rows * sizeof(double *));
  array[0] = (double *)malloc(n_rows * n_cols * sizeof(double));
  for (i = 1; i < n_rows; i++)
  {
    array[i] = array[0] + i * n_cols;
  }
  return array;
}

/**
 *	Initialize arrays A and B with random numbers, and array C with zeros.
 *	Each array is setup as a square block of blck_sz.
 **/
void initialize(double **lA, double **lB, double **lC, int blck_sz)
{
  int i, j;
  double value;
  // Set random values...technically it is already random and this is redundant
  for (i = 0; i < blck_sz; i++)
  {
    for (j = 0; j < blck_sz; j++)
    {
      lA[i][j] = (double)rand() / (double)RAND_MAX;
      lB[i][j] = (double)rand() / (double)RAND_MAX;
      lC[i][j] = 0.0;
    }
  }
}

/**
 *	Perform the SUMMA matrix multiplication.
 *       Follow the pseudo code in lecture slides.
 */
void matmulAdd(double **my_C, double **buffA, double **buffB, int block_sz)
{
  int i, j, k;
  for (k = 0; k < block_sz; k++)
  {
    for (i = 0; i < block_sz; i++)
    {
      for (j = 0; j < block_sz; j++)
      {

        my_C[i][j] += buffA[i][k] * buffB[k][j];
      }
    }
  }
}

void testmatmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
                double **my_B, double **my_C)
{

  // Add your implementation of SUMMA algorithm
  int p, myrank, q;
  MPI_Comm grid_comm;
  int dimsizes[2];
  int wraparound[2];
  int coordinates[2];
  int free_coords[2];
  int reorder = 1;
  int my_grid_rank, grid_rank;
  int row_test, col_test;

  MPI_Comm row_comm;
  MPI_Comm col_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  p = proc_grid_sz * proc_grid_sz;
  // q : 矩陣一邊有幾個
  // q = sqrt(p);
  q = (int)sqrt((int)p);
  dimsizes[0] = dimsizes[1] = q;
  wraparound[0] = wraparound[1] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
  MPI_Comm_rank(grid_comm, &my_grid_rank);
  MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
  MPI_Cart_rank(grid_comm, coordinates, &grid_rank);

  // printf("Process %d > my_grid_rank = %d, coords = (%d, %d), grid_rank =
  // %d\n", 	   myrank, my_grid_rank, coordinates[0], coordinates[1],
  // grid_rank);

  free_coords[0] = 0;
  free_coords[1] = 1;
  MPI_Cart_sub(grid_comm, free_coords, &row_comm);
  if (coordinates[1] == 0)
    row_test = coordinates[0];
  else
    row_test = -1;
  MPI_Bcast(&row_test, 1, MPI_INT, 0, row_comm);
  // printf("Process %d > coords = (%d, %d), row_test = %d\n", myrank,
  //	   coordinates[0], coordinates[1], row_test);

  free_coords[0] = 1;
  free_coords[1] = 0;
  MPI_Cart_sub(grid_comm, free_coords, &col_comm);
  if (coordinates[0] == 0)
    col_test = coordinates[1];
  else
    col_test = -1;
  MPI_Bcast(&col_test, 1, MPI_INT, 0, col_comm);
  // printf("Process %d > coords = (%d, %d), col_test = %d\n", myrank,
  //	   coordinates[0], coordinates[1], col_test);

  // initialize bi/tridiagonal test matrices
  for (int i = 0; i < block_sz; i++)
  {
    for (int j = 0; j < block_sz; j++)
    {
      int global_i = i + block_sz * coordinates[0];
      int global_j = j + block_sz * coordinates[1];
      my_A[i][j] = 0.0;
      my_B[i][j] = 0.0;
      my_C[i][j] = 0.0;
      if (global_i == global_j)
      {
        my_A[i][j] = 1.0;
        my_B[i][j] = 1.0;
      }
      if (global_i == global_j - 1)
      {
        my_B[i][j] = 1.0;
      }
      if (global_i == global_j + 1)
      {
        my_A[i][j] = 1.0;
      }
    }
  }

  double **buffA, **buffB;
  buffA = alloc_2d_double(block_sz, block_sz);
  buffB = alloc_2d_double(block_sz, block_sz);
  for (int k = 0; k < proc_grid_sz; k++)
  {
    // block location , column
    if (coordinates[1] == k)
    {
      memcpy(&buffA[0][0], &my_A[0][0], sizeof(double) * block_sz * block_sz);
    }

    MPI_Bcast(*buffA, block_sz * block_sz, MPI_DOUBLE, k, row_comm);

    if (coordinates[0] == k)
    {
      memcpy(&buffB[0][0], &my_B[0][0], sizeof(double) * block_sz * block_sz);
    }

    MPI_Bcast(*buffB, block_sz * block_sz, MPI_DOUBLE, k, col_comm);

    matmulAdd(my_C, buffA, buffB, block_sz);
  }

  // vaildate sub-bi/tridiagonal-matrix C  computing result
  bool result = true;
  for (int i = 0; i < block_sz; i++)
  {
    for (int j = 0; j < block_sz; j++)
    {
      int global_i = i + block_sz * coordinates[0];
      int global_j = j + block_sz * coordinates[1];
      if ((global_i != 0 && global_j != 0) && global_i == global_j)
      {
        if (my_C[i][j] != 2)
        {
          result = false;
        }
      }
      else if ((global_i == 0 && global_j == 0) || global_i == global_j - 1 ||
               global_i == global_j + 1)
      {
        if (my_C[i][j] != 1)
        {
          result = false;
        }
      }
      else
      {
        if (my_C[i][j] != 0)
        {
          result = false;
        }
      }
    }
  }

  printf("processor %d sub-bi/tridiagonal-matrix validate result is %s \n", my_rank,
         result ? "true" : "false");
}

void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
            double **my_B, double **my_C)
{

  // Add your implementation of SUMMA algorithm
  int p, myrank, q;
  MPI_Comm grid_comm;
  int dimsizes[2];
  int wraparound[2];
  int coordinates[2];
  int free_coords[2];
  int reorder = 1;
  int my_grid_rank, grid_rank;
  int row_test, col_test;

  MPI_Comm row_comm;
  MPI_Comm col_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  p = proc_grid_sz * proc_grid_sz;
  // q : 矩陣一邊有幾個
  // q = sqrt(p);
  q = (int)sqrt((int)p);
  dimsizes[0] = dimsizes[1] = q;
  wraparound[0] = wraparound[1] = 1;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
  MPI_Comm_rank(grid_comm, &my_grid_rank);
  MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
  MPI_Cart_rank(grid_comm, coordinates, &grid_rank);

  // printf("Process %d > my_grid_rank = %d, coords = (%d, %d), grid_rank =
  // %d\n", 	   myrank, my_grid_rank, coordinates[0], coordinates[1],
  // grid_rank);

  free_coords[0] = 0;
  free_coords[1] = 1;
  MPI_Cart_sub(grid_comm, free_coords, &row_comm);
  if (coordinates[1] == 0)
    row_test = coordinates[0];
  else
    row_test = -1;
  MPI_Bcast(&row_test, 1, MPI_INT, 0, row_comm);
  // printf("Process %d > coords = (%d, %d), row_test = %d\n", myrank,
  //	   coordinates[0], coordinates[1], row_test);

  free_coords[0] = 1;
  free_coords[1] = 0;
  MPI_Cart_sub(grid_comm, free_coords, &col_comm);
  if (coordinates[0] == 0)
    col_test = coordinates[1];
  else
    col_test = -1;
  MPI_Bcast(&col_test, 1, MPI_INT, 0, col_comm);
  // printf("Process %d > coords = (%d, %d), col_test = %d\n", myrank,
  //	   coordinates[0], coordinates[1], col_test);

  double **buffA, **buffB;
  buffA = alloc_2d_double(block_sz, block_sz);
  buffB = alloc_2d_double(block_sz, block_sz);
  //proc_grid_sz = (int)sqrt((int)num_proc);
  for (int k = 0; k < proc_grid_sz; k++)
  {
    // block location , column
    if (coordinates[1] == k)
    {
      memcpy(&buffA[0][0], &my_A[0][0], sizeof(double) * block_sz * block_sz);
    }

    MPI_Bcast(*buffA, block_sz * block_sz, MPI_DOUBLE, k, row_comm);

    if (coordinates[0] == k)
    {
      memcpy(&buffB[0][0], &my_B[0][0], sizeof(double) * block_sz * block_sz);
    }

    MPI_Bcast(*buffB, block_sz * block_sz, MPI_DOUBLE, k, col_comm);

    matmulAdd(my_C, buffA, buffB, block_sz);
  }
}

int main(int argc, char *argv[])
{
  int rank, num_proc;                      // process rank and total number of processes
  double start_time, end_time, total_time; // for timing
  int block_sz;                            // Block size length for each processor to handle
  int proc_grid_sz;                        // 'q' from the slides

  srand(time(NULL)); // Seed random numbers

  /* insert MPI functions to 1) start process, 2) get total number of processors
   * and 3) process rank*/

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  /* assign values to 1) proc_grid_sz and 2) block_sz*/
  proc_grid_sz = (int)sqrt((int)num_proc);
  // proc_grid_sz = sqrt(num_proc);
  block_sz = SZ / proc_grid_sz;
  if (SZ % proc_grid_sz != 0)
  {
    printf("Matrix size cannot be evenly split amongst resources!\n");
    printf("Quitting....\n");
    exit(-1);
  }

  // Create the local matrices on each process

  double **A, **B, **C;
  A = alloc_2d_double(block_sz, block_sz);
  B = alloc_2d_double(block_sz, block_sz);
  C = alloc_2d_double(block_sz, block_sz);

  initialize(A, B, C, block_sz);

  // Use MPI_Wtime to get the starting time
  // printf("process %d start computing and timeing \n", rank);
  start_time = MPI_Wtime();

  // Use SUMMA algorithm to calculate product C

  matmul(rank, proc_grid_sz, block_sz, A, B, C);

  // printf("process %d End computing and timeing\n", rank);
  // Use MPI_Wtime to get the finishing time

  MPI_Barrier(MPI_COMM_WORLD);

  end_time = MPI_Wtime();
  // Obtain the elapsed time and assign it to total_time
  total_time = end_time - start_time;

  // print submatrix A , B , C to check
  /* 	

  printf("I am process %d  start printing A \n", rank);
  printmatrix(A,block_sz);

  printf("I am process %d  start printing B \n", rank);
  printmatrix(B,block_sz);


  printf("I am process %d  start printing C \n", rank);
  printmatrix(C,block_sz);

  */

  // Insert statements for testing
  testmatmul(rank, proc_grid_sz, block_sz, A, B, C);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
  {
    // Print in pseudo csv format for easier results compilation

    printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n", SZ,
           num_proc, total_time);
  }

  // Destroy MPI processes

  MPI_Finalize();
  return 0;
}
