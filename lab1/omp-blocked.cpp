#include <cstring>
#include <iostream>
#include "gemm.h"

// Using declarations, if any...
const int finalSize = kI * kJ;

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ])
{
  int k, j, ii, ik;
  float sum;
  const int BLOCK_SIZE = 32;
  float aV;
  float *cp = (float *)c;
  std::memset(cp, 0, sizeof(float) * kJ * kI);
#pragma omp parallel for private(k, j, ii, ik, sum) schedule(static, 32)
  for (int i = 0; i < kI; i += BLOCK_SIZE)
  {
    for (k = 0; k < kK; k += BLOCK_SIZE)
    {
      for (ii = i; ii < i + BLOCK_SIZE; ++ii)
      {
        for (ik = k; ik < k + BLOCK_SIZE; ++ik)
        {
          //#pragma omp simd
          for (j = 0; j < kJ; j++)
          {
            c[ii][j] += a[ii][ik] * b[ik][j];
          }
        }
      }
    }
  }
}
