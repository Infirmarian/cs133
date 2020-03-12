const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

#define max(X,Y) ((X)>(Y)?(X):(Y))

#pragma ACCEL kernel
void CnnKernel(
    const float input[kNum][kInImSize][kInImSize],
    const float weight[kNum][kNum][kKernel][kKernel], const float bias[kNum],
    float output[kNum][kOutImSize][kOutImSize]) {

  float C[kImSize][kImSize];

  for (int i = 0; i < kNum; ++i) {
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        C[h][w] = bias[i];
    }

  // Convolution
    for (int j = 0; j < kNum; ++j) {
      for (int h = 0; h < kImSize; ++h) {
        for (int w = 0; w < kImSize; ++w) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q)
              C[h][w] += weight[i][j][p][q] * input[j][h + p][w + q];
          }
        }
      }
    }

  // ReLU
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C[h][w] = max(0.f, C[h][w]);
      }
    }

  // Max pooling
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        output[i][h][w] = max(
            max(C[h * 2][w * 2    ], C[h * 2 + 1][w * 2    ]),
            max(C[h * 2][w * 2 + 1], C[h * 2 + 1][w * 2 + 1]));
      }
    }
  }
}
  