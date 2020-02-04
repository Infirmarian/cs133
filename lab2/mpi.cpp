#include </usr/include/mpi/mpi.h>
#include <iostream>
#include <cstring>
#include "../lab1/gemm.h"
#define ROOT 0
// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  float *A = (float *)malloc(kI * kK * sizeof(float));
  float *B = (float *)malloc(kI * kK * sizeof(float));
  if (rank == ROOT)
  {
    std::memcpy(A, a, kI * kK);
    std::memcpy(B, b, kI * kK);
    std::cout << "Copied all memory\n";
  }
  MPI_Bcast((void *)A, kI * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)B, kI * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  std::cout << "Sent data\nA[0][0] = " << A[0] << std::endl;
  //    MPI_Bcast((void *)b, kI * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);
  //    MPI_Bcast((void *)c, kI * kK, MPI_FLOAT, 0, MPI_COMM_WORLD);

  {
    //    MPI_Status s;
    //    MPI_Recv((void *)A, kI * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &s);
    //    std::cout << "Rank " << rank << ": " << s.MPI_ERROR << std::endl;
  }
  //  if (rank == 1)
  //  {
  // MPI_Status s;
  // float *A = (float *)malloc(kI * kK * sizeof(float));
  //    float *B = (float *)malloc(kI * kK * sizeof(float));
  //    float *C = (float *)malloc(kI * kK * sizeof(float));
  //    MPI_Recv((void *)B, kI * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &s);
  //    MPI_Recv((void *)C, kI * kK, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &s);
  /* for (int i = 0; i < kI; i++)
    {
      for (int k = 0; k < kK; k++)
      {
        for (int j = 0; j < kJ; j++)
        {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }*/
  //}
}
