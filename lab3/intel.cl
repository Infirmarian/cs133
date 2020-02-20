__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;
__constant int TJ = 8;
__constant int TW = 8;
__constant int alignment = 1024;

__kernel void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
  __local float C[kImSize], C2[kImSize];
  int id = get_global_id(0);
  int size = get_global_size(0);

  for (int i = id; i < kNum; i+=size) {
    float4 biasv = (float4)(bias[i], bias[i], bias[i], bias[i]);
    for (int h = 0; h < kImSize; h+=2) {
      for (int w = 0; w < kImSize; ++w){
        C[w] = bias[i];
        C2[w] = bias[i];
      }
    for (int j = 0; j < kNum; j+=TJ) {
      for (int w = 0; w < kImSize; w+=4) {
        for(int jj = j; jj < j + TJ; ++jj){
            for (int p = 0; p < kKernel; ++p) {
              for (int q = 0; q < kKernel; ++q){
                float where = weight[i*kKernel*kKernel*kNum + jj*kKernel*kKernel+ p*kKernel + q];
                __constant float* c1temp = input + (jj*kInImSize*kInImSize + kInImSize*(h + p) + w+ q);
                __constant float* c2temp = input + (jj*kInImSize*kInImSize + kInImSize*(h + 1+ p) + w+ q);
                C[w] += where * *c1temp;
                C[w+1] += where * *(++c1temp);
                C[w+2] += where * *(++c1temp);
                C[w+3] += where * *(++c1temp);
                C2[w] += where * *c2temp;
                C2[w+1] += where * *(++c2temp);
                C2[w+2] += where * *(++c2temp);
                C2[w+3] += where * *(++c2temp);
              }
            }
          
        }
      }
    }
    for (int w = 0; w < kOutImSize; ++w) {
      for(int ww = 0; ww < 1; ++ww){
        output[i*kOutImSize*kOutImSize + (h/2)*kOutImSize + w] = max (0.f, max(
            max(C[w * 2 ], C2[ w * 2    ]),
            max(C[w * 2 + 1], C2[w * 2 + 1])));
      }
    }
  }
  }
}
