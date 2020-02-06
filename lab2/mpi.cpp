#include </usr/include/mpi/mpi.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include "../lab1/gemm.h"
#define ROOT 0
#define I_BLOCK_SIZE 64
#define J_BLOCK_SIZE 1024
#define K_BLOCK_SIZE 8
void Gemm(const float *a, const float *b, float *c, int lI, int lJ, int lK)
{
  // std::cout << lI << " " << lJ << " " << lK << std::endl;
  float temp;
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
            temp = c[ii * lJ + jj];
            for (int kk = k; kk < k + K_BLOCK_SIZE; ++kk)
            {
              temp += a[ii * lK + kk] * b[kk * lJ + jj];
            }
            c[ii * lJ + jj] = temp;
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
  // int horizontalChunkSize = kK / floor(sqrt(size));
  // int verticalChunkSize = kJ / (ceil(sqrt(size)));
  // int horizontalCount = kK / horizontalChunkSize;
  // int verticalCount = kJ / verticalChunkSize;
  float *A = (float *)aligned_alloc(64, sizeof(float) * stripSize * kK);
  float *B = (float *)aligned_alloc(64, sizeof(float) * kJ * kK);
  float *C = (float *)aligned_alloc(64, sizeof(float) * stripSize * kJ);
  memset(C, 0, sizeof(float) * stripSize * kJ);
  MPI_Scatter(a, kK * stripSize, MPI_FLOAT, A, kK * stripSize, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  if (rank == ROOT)
    memcpy(B, b, sizeof(float) * kJ * kK);
  MPI_Bcast(B, kJ * kK, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  Gemm(A, B, C, stripSize, kJ, kK);
  MPI_Gather(C, stripSize * kJ, MPI_FLOAT, c, stripSize * kJ, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
  // if (rank == ROOT)
  // {
  //   float *A = (float *)a;
  //   int hoffset, voffset;
  //   for (int i = size - 1; i > 0; i--)
  //   {
  //     hoffset = horizontalChunkSize * ((size - i) / verticalCount);
  //     voffset = verticalChunkSize * (i / horizontalCount);
  //     std::cout << "Rank " << i << ": H offset: " << hoffset << " V offset: " << voffset << std::endl;
  //     MPI_Send(A + (kK * hoffset), horizontalChunkSize * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
  //     // Permute the B matrix to send data
  //     for (int k = 0; k < kK; k++)
  //     {
  //       for (int j = 0; j < verticalChunkSize; j++)
  //       {
  //         B[j * kK + k] = b[k][j + voffset];
  //       }
  //     }
  //     MPI_Send(B, verticalChunkSize * kK, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
  //   }
  //   std::cout << "Transmitted all data\n";
  //   //    float *newB = (float *)aligned_alloc(64, sizeof(float) * kK * kJ);
  //   for (int k = 0; k < kK; k++)
  //   {
  //     for (int j = 0; j < verticalChunkSize; j++)
  //     {
  //       B[j * kK + k] = b[k][j];
  //     }
  //   }
  //   std::cout << "Calculating..." << std::endl;
  //   memset(C, 0, horizontalChunkSize * verticalChunkSize);
  //   GemmT((float *)a, B, (float *)c, horizontalChunkSize, verticalChunkSize, kK);
  //   GemmT((float *)a, B, C, horizontalChunkSize, verticalChunkSize, kK);

  //   for (int j = 0; j < horizontalChunkSize; j++)
  //   {
  //     for (int k = 0; k < verticalChunkSize; k++)
  //     {
  //       c[k][j] = C[j * verticalChunkSize + k];
  //     }
  //   }
  // }
  // else
  // {
  //   MPI_Status s;
  //   MPI_Recv(A, horizontalChunkSize * kK, MPI_FLOAT, ROOT, 0, MPI_COMM_WORLD, &s);
  //   MPI_Recv(B, verticalChunkSize * kK, MPI_FLOAT, ROOT, 0, MPI_COMM_WORLD, &s);
  //   GemmT(A, B, C, horizontalChunkSize, verticalChunkSize, kK);
  //   std::cout << "Done calculating from " << rank << std::endl;
  //   MPI_Send(C, horizontalChunkSize * verticalChunkSize, MPI_FLOAT, ROOT, 0, MPI_COMM_WORLD);
  // }
  // if (rank == ROOT)
  // {
  //   for (int i = 1; i < size; i++)
  //   {
  //     MPI_Status s;
  //     MPI_Recv(C, horizontalChunkSize * verticalChunkSize, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &s);
  //     int hoffset = horizontalChunkSize * ((size - i) / verticalCount);
  //     int voffset = verticalChunkSize * (i / horizontalCount);
  //     std::cout << "Rank " << i << " has vertical " << voffset << " and horizontal " << hoffset << std::endl;
  //     std::cout << "Calculated: " << C[0] << " Actual: " << c[hoffset][voffset] << std::endl;
  //     std::cout << "Calculated: " << C[1] << " Actual: " << c[hoffset][voffset + 1] << std::endl;
  //     for (int j = 0; j < horizontalChunkSize; j++)
  //     {
  //       //        std::cout << c[j + hoffset][voffset] << " " << C[verticalChunkSize * j] << std::endl;
  //       std::memcpy(c[j + hoffset] + voffset, C + (verticalChunkSize * j), verticalChunkSize);
  //     }
  //     std::cout << "Received from " << i << std::endl;
  //   }
  // }
}
