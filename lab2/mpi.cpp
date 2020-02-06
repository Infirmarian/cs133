#include </usr/include/mpi/mpi.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include "../lab1/gemm.h"
#define ROOT 0

void GemmT(const float *a, const float *b, float *c, int I, int J, int K)
{
  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < J; j++)
    {
      for (int k = 0; k < K; k++)
      {
        c[i * J + j] += a[i * K + k] * b[j * K + k];
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
  // if (rank == ROOT)
  // {
  //   std::cout << horizontalChunkSize << std::endl;
  //   std::cout << verticalChunkSize << std::endl;
  //   std::cout << horizontalCount << std::endl;
  //   std::cout << verticalCount << std::endl;
  // }
  float *A = (float *)aligned_alloc(64, sizeof(float) * horizontalChunkSize * kK);
  float *B = (float *)aligned_alloc(64, sizeof(float) * verticalChunkSize * kK);
  float *C = (float *)aligned_alloc(64, sizeof(float) * horizontalChunkSize * verticalChunkSize);
  if (rank == ROOT)
  {
    float *A = (float *)a;
    int hoffset, voffset;
    for (int i = size - 1; i > 0; i--)
    {
      hoffset = horizontalChunkSize * ((size - i) / verticalCount);
      voffset = verticalChunkSize * (i / horizontalCount);
      std::cout << "Rank " << i << ": H offset: " << hoffset << " V offset: " << voffset << std::endl;
      MPI_Send(A + (kK * hoffset), horizontalChunkSize * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      // Permute the B matrix to send data
      for (int k = 0; k < kK; k++)
      {
        for (int j = 0; j < verticalChunkSize; j++)
        {
          B[j * kK + k] = b[k][j + voffset];
        }
      }
      MPI_Send(B, verticalChunkSize * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
    std::cout << "Transmitted all data\n";
    float *newB = (float *)aligned_alloc(64, sizeof(float) * kK * kJ);
    for (int k = 0; k < kK; k++)
    {
      for (int j = 0; j < kJ; j++)
      {
        newB[j * kK + k] = b[k][j];
      }
    }
    std::cout << "Calculating>>>" << std::endl;
    GemmT((float *)a, (float *)newB, (float *)c, kI, kJ, kK);
  }
  else
  {
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
      int hoffset = horizontalChunkSize * ((size - i) / verticalCount);
      int voffset = verticalChunkSize * (i / horizontalCount);
      std::cout << "Rank " << i << " has vertical " << voffset << " and horizontal " << hoffset << std::endl;
      std::cout << "Calculated: " << C[0] << " Actual: " << c[hoffset][voffset] << std::endl;
      std::cout << "Calculated: " << C[1] << " Actual: " << c[hoffset][voffset + 1] << std::endl;
      for (int j = 0; j < horizontalChunkSize; j++)
      {
        //        std::cout << c[j + hoffset][voffset] << " " << C[verticalChunkSize * j] << std::endl;
        std::memcpy(c[j + hoffset] + voffset, C + (verticalChunkSize * j), verticalChunkSize);
      }
      std::cout << "Received from " << i << std::endl;
    }
  }
  // free(A);
  // free(B);
}
