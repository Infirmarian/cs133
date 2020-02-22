__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;
__constant int TJ = 2;
__constant int TW = 1;

#define kVectorImSize 14 //kImSize/16;
__kernel void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {
  // your code goes here
  __private float16 C[kVectorImSize], C2[kVectorImSize];
  const float16 zeros = (float16)(0);
  int i = get_global_id(0);
  int hv = get_global_id(1)<<1;
  int size = get_global_size(0);
  int kInImSizeSquare = kInImSize*kInImSize;
  C[0] = C[1] = C[2] = C[3] = C[4] = C[5] = C[6] = C[7] = \
  C[8] = C[9] = C[10] = C[11] = C[12] = C[13] = (float16)(bias[i]);
  C2[0] = C2[1] = C2[2] = C2[3] = C2[4] = C2[5] = C2[6] = C2[7] = \
  C2[8] = C2[9] = C2[10] = C2[11] = C2[12] =  C2[13] = (float16)(bias[i]);
  for (int w = 0; w < kVectorImSize; w+=TW) {
    for (int j = 0; j < kNum; j+=TJ) {
      float16 t1 = zeros;
      float16 t2 = zeros;
      for (int p = 0; p < kKernel; ++p) {
        __constant float* wBase = weight + i*kKernel*kKernel*kNum + j*kKernel*kKernel+ p*kKernel;
        __constant float* wBase2 = wBase + kKernel*kKernel;
        __constant float* cptemp = input + (j*kInImSizeSquare + kInImSize*(hv + p) + (w<<4));
        __constant float* cptempKIM = cptemp + kInImSize;
        t1 += *(wBase) * vload16(0, cptemp) + *(wBase2) *vload16(0, cptemp + kInImSizeSquare) +
                  *(wBase+1) * vload16(0, cptemp+1) + *(wBase2+1) *vload16(0, cptemp + kInImSizeSquare+1) +
                  *(wBase+2) * vload16(0, cptemp+2) + *(wBase2+2) *vload16(0, cptemp + kInImSizeSquare+2) +
                  *(wBase+3) * vload16(0, cptemp+3) + *(wBase2+3) *vload16(0, cptemp + kInImSizeSquare+3) +
                  *(wBase+4) * vload16(0, cptemp+4) + *(wBase2+4) *vload16(0, cptemp + kInImSizeSquare+4);
        t2 += *(wBase) * vload16(0, cptempKIM) + *(wBase2) * vload16(0, cptempKIM+kInImSizeSquare) +
                  *(wBase+1) * vload16(0, cptempKIM+1) + *(wBase2+1) * vload16(0, cptempKIM+kInImSizeSquare +1) +
                  *(wBase+2) * vload16(0, cptempKIM+2) + *(wBase2+2) * vload16(0, cptempKIM+kInImSizeSquare +2) +
                  *(wBase+3) * vload16(0, cptempKIM+3) + *(wBase2+3) * vload16(0, cptempKIM+kInImSizeSquare +3) +
                  *(wBase+4) * vload16(0, cptempKIM+4) + *(wBase2+4) * vload16(0, cptempKIM+kInImSizeSquare +4);
      }
      *(C+w) += t1;
      *(C2+w) += t2;
    }
  }
//  for(int w = 0; w<kVectorImSize; w+=2)
//  {
    C[0] = max(max(C2[0], C[0]), zeros);
    C[1] = max(max(C2[1], C[1]), zeros);
    C[2] = max(max(C2[2], C[2]), zeros);
    C[3] = max(max(C2[3], C[3]), zeros);
    C[4] = max(max(C2[4], C[4]), zeros);
    C[5] = max(max(C2[5], C[5]), zeros);
    C[6] = max(max(C2[6], C[6]), zeros);
    C[7] = max(max(C2[7], C[7]), zeros);
    C[8] = max(max(C2[8], C[8]), zeros);
    C[9] = max(max(C2[9], C[9]), zeros);
    C[10] = max(max(C2[10], C[10]), zeros);
    C[11] = max(max(C2[11], C[11]), zeros);
    C[12] = max(max(C2[12], C[12]), zeros);
    C[13] = max(max(C2[13], C[13]), zeros);
//  }
  __private float* cO = (__private float*) C;
  output += i*kOutImSize*kOutImSize + (hv>>1)*kOutImSize;
  for (int w = 0; w < kOutImSize; ++w) {
      *(output+w) = max (*(cO++), *(cO++));
    }
}