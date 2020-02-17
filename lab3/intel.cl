__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__kernel void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
  local float C[kImSize], C2[kImSize];
  int id = get_global_id(0);
  int size = get_global_size(0);

  printf("ID: %d, Size: %d\n", id, size);
  for (int i = id; i < kNum; i+=size) {
    for (int h = 0; h < kImSize; h+=2) {
      for (int w = 0; w < kImSize; ++w){
        C[w] = bias[i];
        C2[w] = bias[i];
      }
      for (int j = 0; j < kNum; ++j) {
          for (int w = 0; w < kImSize; ++w) {
            for (int p = 0; p < kKernel; ++p) {
              for (int q = 0; q < kKernel; ++q){
                C[w] += weight[i*kKernel*kKernel*kNum + j*kKernel*kKernel+ p*kKernel + q] *\
                input[j*kInImSize*kInImSize + kInImSize*(h + p) + w + q];
                C2[w] += weight[i*kKernel*kKernel*kNum + j*kKernel*kKernel+ p*kKernel + q] *\
                input[j*kInImSize*kInImSize + kInImSize*(h+1 + p) + w + q];
              }
            }
          }
      }
      for (int w = 0; w < kImSize; ++w) {
        C[w] = max(0.f, C[w]);
        C2[w] = max(0.f, C2[w]);
      }
      for (int w = 0; w < kOutImSize; ++w) {
        output[i*kOutImSize*kOutImSize + (h/2)*kOutImSize + w] = max(
            max(C[w * 2  ], C2[ w * 2    ]),
            max(C[w * 2 + 1], C2[w * 2 + 1]));
      }
  }
  }
}
