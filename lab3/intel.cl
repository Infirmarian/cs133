__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

__constant int alignment = 2048;
__kernel void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
  __private float ualC[kImSize + alignment], ualC2[kImSize + alignment];
  int offset = (int)ualC % alignment;
  int offset2 = (int)ualC2 % alignment;
  __private float* C = ualC + alignment - offset;
  __private float* C2 = ualC2 + alignment - offset2;
  int i = get_global_id(0);
  int h = get_global_id(1)*2;
  int size = get_global_size(0);

//  printf("G1: %d, G2: %d\n", i, h);
  for (int w = 0; w < kImSize; ++w){
    C[w] = bias[i];
    C2[w] = bias[i];
  }
  float l;
  for (int j = 0; j < kNum; ++j) {
    for (int w = 0; w < kImSize; ++w) {
      for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q){
          l = weight[i*kKernel*kKernel*kNum + j*kKernel*kKernel+ p*kKernel + q];
          C[w] += l * input[j*kInImSize*kInImSize + kInImSize*(h + p) + w + q];
          C2[w] += l * input[j*kInImSize*kInImSize + kInImSize*(h+1 + p) + w + q];
        }
      }
    }
  }
  for (int w = 0; w < kOutImSize; ++w) {
    output[i*kOutImSize*kOutImSize + (h/2)*kOutImSize + w] = max(0.f, max(
        max(C[w * 2  ], C2[ w * 2    ]),
        max(C[w * 2 + 1], C2[w * 2 + 1])));
  }
}
