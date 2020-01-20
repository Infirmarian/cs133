// Header inclusions, if any...
#include <cstring>

#include "gemm.h"

const int finalSize = kI * kK;
void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ])
{
  int k, j;
  float *cp = (float *)c;
  std::memset(cp, 0, sizeof(float) * kJ * kI);
#pragma omp parallel for private(k, j)
  for (int i = 0; i < kI; ++i)
  {
    for (k = 0; k < kK; ++k)
    {
      for (j = 0; j < kJ; ++j)
      {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}
