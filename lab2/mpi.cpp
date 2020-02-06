#include </usr/include/mpi/mpi.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include "../lab1/gemm.h"
#define ROOT 0
// Using declarations, if any...

void Gemm(const float *a, const float *b, float *c, int I, int J, int K)
{
  for (int i = 0; i < I; i++)
  {
    for (int k = 0; k < K; k++)
    {
      for (int j = 0; j < J; j++)
      {
        c[i * J + j] += a[i * K + k] * b[k * J + j];
      }
    }
  }
}
void GemmT(const float *a, const float *b, float *c, int I, int J, int K)
{
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      for (int k = 0; k < K; k++)
      {
        c[i * I + j] += a[i * I + k] * b[j * J + k];
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
  int horizontalChunkSize = kK / floor(sqrt(size));
  int verticalChunkSize = kJ / (ceil(sqrt(size)));
  int horizontalCount = kK / horizontalChunkSize;
  int verticalCount = kJ / verticalChunkSize;
  if (rank == ROOT)
  {
    std::cout << horizontalChunkSize << std::endl;
    std::cout << verticalChunkSize << std::endl;
    std::cout << horizontalCount << std::endl;
    std::cout << verticalCount << std::endl;
  }
  //  std::cout << "Vertical chunk size: " << verticalChunkSize << "\nHorizontal chunk size: " << horizontalChunkSize << std::endl;
  float *A = (float *)aligned_alloc(64, sizeof(float) * horizontalChunkSize * kI);
  float *B = (float *)aligned_alloc(64, sizeof(float) * verticalChunkSize * kJ);
  float *C = (float *)aligned_alloc(64, sizeof(float) * horizontalChunkSize * verticalChunkSize);
  //  std::cout << "A: " << A << std::endl;
  //  std::cout << "B: " << B << std::endl;
  if (rank == ROOT)
  {
    float *A = (float *)a;
    for (int i = 0; i < verticalChunkSize; i++)
    {
      // TODO: Get some pencil and paper and work this boy out!
    }
    std::cout << "Preparing to transmit portions of the data matrix\n";
    for (int i = 1; i < size; i++)
    {
      int hoffset = horizontalChunkSize * (i / verticalCount);
      int voffset = verticalChunkSize * (i / horizontalCount);
      std::cout << "H offset: " << hoffset << " V offset: " << voffset << std::endl;
      MPI_Send(A + (kK * hoffset), horizontalChunkSize * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      // Permute the B matrix to send data
      for (int j = 0; j < verticalChunkSize; j++)
      {
        for (int k = 0; k < kK; k++)
        {
          B[j * kK + k] = b[j][k];
        }
      }
      MPI_Send(B, verticalChunkSize * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      std::cout << "Sent to " << i << std::endl;
    }
    std::cout << "Transmitted all data\n";
    float *newB = (float *)aligned_alloc(64, sizeof(float) * kK * kJ);
    for (int k = 0; k < kK; k++)
    {
      for (int j = 0; j < kJ; j++)
      {

        newB[j * kJ + k] = b[k][j];
      }
    }
    GemmT((float *)a, (float *)newB, (float *)c, kI, kJ, kK);
  }
  else
  {
    //    std::cout << "Receiving\n";
    MPI_Status s;
    MPI_Recv(A, horizontalChunkSize * kK, MPI_FLOAT, ROOT, 0, MPI_COMM_WORLD, &s);
    MPI_Recv(B, verticalChunkSize * kK, MPI_FLOAT, ROOT, 0, MPI_COMM_WORLD, &s);
    GemmT(A, B, C, horizontalChunkSize, verticalChunkSize, kK);
    std::cout << "Done calculating from " << rank << std::endl;
    MPI_Send(C, horizontalChunkSize * verticalChunkSize, MPI_FLOAT, ROOT, 0, MPI_COMM_WORLD);
  }
  if (rank == ROOT)
  {
    for (int i = 1; i < size; i++)
    {
      MPI_Status s;
      MPI_Recv(C, horizontalChunkSize * verticalChunkSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &s);
      std::cout << "Received from " << i << std::endl;
    }
  }
  // free(A);
  // free(B);
}
