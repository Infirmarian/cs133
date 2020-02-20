__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;
__constant int TJ = 8;
__constant int TW = 8;
__constant int alignment = 1024;

__constant int kVectorImSize = kImSize/4;
__kernel void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
  __local float4 C[kImSize], C2[kImSize];
  int id = get_global_id(0);
  int size = get_global_size(0);

  for (int i = id; i < kNum; i+=size) {
    float4 biasv = (float4)(bias[i], bias[i], bias[i], bias[i]);
    for (int h = 0; h < kImSize; h+=2) {
      for (int w = 0; w < kVectorImSize; ++w){
        C[w] = biasv;
        C2[w] = biasv;
      }
    float where;
    for (int j = 0; j < kNum; j+=TJ) {
      for (int w = 0; w < kVectorImSize; w++) {
        for(int jj = j; jj < j + TJ; ++jj){
            for (int p = 0; p < kKernel; ++p) {
              for (int q = 0; q < kKernel; ++q){
                where = weight[i*kKernel*kKernel*kNum + jj*kKernel*kKernel+ p*kKernel + q];
                __constant float* c1temp = input + (jj*kInImSize*kInImSize + kInImSize*(h + p) + 4*w+ q);
                __constant float* c2temp = input + (jj*kInImSize*kInImSize + kInImSize*(h + 1+ p) + 4*w+ q);
                float4 c1a = where * (float4)(*c1temp, *(c1temp+1), *(c1temp+2), *(c1temp+3));
                float4 c2a = where * (float4)(*c2temp, *(c2temp+1), *(c2temp+2), *(c2temp+3));
                C[w] += c1a;
                C2[w] += c2a;
              }
            }
          
        }
      }
    }
    __local float* cO = (__local float*) C;
    __local float* cO2 = (__local float*) C2;
    for (int w = 0; w < kOutImSize; ++w) {
        output[i*kOutImSize*kOutImSize + (h/2)*kOutImSize + w] = max (0.f, max(
            max(cO[w *2], cO2[ w * 2    ]),
            max(cO[w *2 + 1], cO2[w * 2 + 1])));
      }
    }
  }
}
