#include </usr/include/mpi/mpi.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include "../lab1/gemm.h"
#define ROOT 0
#define I_BLOCK_SIZE 64
#define J_BLOCK_SIZE 1024
#define K_BLOCK_SIZE 8
void Gemm(const float *a, const float *b, float *c, int lI, int lJ, int lK)
{
  int aind, bind;
  for (int i = 0; i < lI; i += I_BLOCK_SIZE)
  {
    for (int k = 0; k < lK; k += K_BLOCK_SIZE)
    {
      for (int j = 0; j < lJ; j += J_BLOCK_SIZE)
      {
        for (int ii = i; ii < i + I_BLOCK_SIZE; ++ii)
        {
          for (int jj = j; jj < j + J_BLOCK_SIZE; ++jj)
          {
            aind = ii * lK + k;
            bind = k * lJ + jj;
            // Unrolled equivelant of loop over k to k+K_BLOCK_SIZE
            c[ii * lJ + jj] +=
                a[aind] * b[bind] + a[aind + 1] * b[bind + lJ] + a[aind + 2] * b[bind + (lJ << 1)] + a[aind + 3] * b[bind + (lJ << 1) + lJ] + a[aind + 4] * b[bind + (lJ << 2)] + a[aind + 5] * b[bind + (lJ << 2) + lJ] + a[aind + 6] * b[bind + (lJ << 2) + (lJ << 1)] + a[aind + 7] * b[bind + (lJ << 3) - lJ];
          }
        }
      }
    }
  }
}

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int stripSize = kI / size;
  float *A = (float *)aligned_alloc(2028, sizeof(float) * stripSize * kK);
  float *B = (float *)aligned_alloc(2028, sizeof(float) * kJ * kK);
  float *C = (float *)aligned_alloc(2028, sizeof(float) * stripSize * kJ);
  memset(C, 0, sizeof(float) * stripSize * kJ);
  MPI_Scatter(a, kK * stripSize, MPI_FLOAT, A, kK * stripSize, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  if (rank == ROOT)
    memcpy(B, b, sizeof(float) * kJ * kK);
  MPI_Bcast(B, kJ * kK, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  Gemm(A, B, C, stripSize, kJ, kK);
  MPI_Gather(C, stripSize * kJ, MPI_FLOAT, c, stripSize * kJ, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
}
