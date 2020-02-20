__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;
__constant int TJ = 2;
__constant int TW = 2;

__constant int kVectorImSize = kImSize/16;
__kernel void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
  __local float16 C[kVectorImSize], C2[kVectorImSize];
  int id = get_global_id(0);
  int hv = get_global_id(1)*2;
  int size = get_global_size(0);

  for (int i = id; i < kNum; i+=size) {
    float16 biasv = (float16)(bias[i]);
//    for (int h = 0; h < kImSize; h+=2) {
      for (int w = 0; w < kVectorImSize; ++w){
        C[w] = biasv;
        C2[w] = biasv;
      }
    float where;
    for (int j = 0; j < kNum; j+=TJ) {
        for(int jj = j; jj < j + TJ; ++jj){
            for (int p = 0; p < kKernel; ++p) {
              for (int w = 0; w < kVectorImSize; w+=TW) {
                for (int q = 0; q < kKernel; ++q){
                  where = weight[i*kKernel*kKernel*kNum + jj*kKernel*kKernel+ p*kKernel + q];
                  for(int ww = w; ww < w + TW; ++ww){
                    __constant float* cptemp = input + (jj*kInImSize*kInImSize + kInImSize*(hv + p) + (ww<<4)+ q);
                    *(C+ww) += where * vload16(0, cptemp);
                    *(C2+ww) += where * vload16(0, cptemp+kInImSize);
                  }
                }
              }
            }
          }
        }
    float16 zeros = (float16)(0);
    for(int w = 0; w<kVectorImSize; w++)
    {
      C[w] = max(max(C2[w], C[w]), zeros);
    }
    __local float* cO = (__local float*) C;
    for (int w = 0; w < kOutImSize; ++w) {
        output[i*kOutImSize*kOutImSize + (hv/2)*kOutImSize + w] = max (*cO, *(cO+1));
        cO += 2;
      }
    }
//  }
}
