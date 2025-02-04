__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

#define input(j, h, w) \
   input[((j) * kInImSize * kInImSize + (h) * kInImSize + (w))]

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {

 float output_buf[kImSize][kImSize] // buffer of output
 ;
 float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
 __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
 __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
 ;

 float weight_buf[kKernel][kKernel] //buffer of weight
 __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
 __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
 ;
for(int i = 0; i<kNum; ++i){
//copy bias here
  copy_bias:
  for (int h = 0; h < kImSize; ++h) {
    __attribute__((xcl_pipeline_loop))
    for (int w = 0; w < kImSize; ++w)
      output_buf[h][w] = bias[i];
  }
  for(int j = 0; j<kNum; ++j){
      //input load loop
    load_in:
    for(int h = 0; h<kInImSize; ++h){
      __attribute__((xcl_pipeline_loop))
      for (int w = 0; w < kInImSize; w ++) {
        for (int q = 0; q < kKernel; ++q) { //make kKernel copy of input(j,h,w)
          input_buf[h][w - q + kKernel - 1][q] = input(j, h, w);
        }
      }
    }
    //copy weight here
    load_weight:
    __attribute__((xcl_pipeline_loop))
    for (int p = 0; p < kKernel; ++p) {
      for (int q = 0; q < kKernel; ++q){
        weight_buf[p][q] = weight[i*kNum*kKernel*kKernel + j*kKernel*kKernel + p*kKernel + q];
      }
    }
    for(int h = 0; h< kImSize; ++h){
      //convolution loop
      conv:
      __attribute__((xcl_pipeline_loop))
      for (int w = 0; w < kImSize; ++w) { //pipelined loop
        float tmp = 0; 
        for (int p = 0; p < kKernel; ++p) {  // unrolled loop
          __attribute__((xcl_pipeline_loop))
         innerQ: for (int q = 0; q < kKernel; ++q) {  //unrolled loop
            tmp += weight_buf[p][q] * input_buf[h + p][w + kKernel - 1][q];
         }
        }
        output_buf[h][w] += tmp; //store reduction result
      }
    }
  }
  relu:
  for (int h = 0; h < kImSize; ++h) {
    for (int w = 0; w < kImSize; ++w) {
      output_buf[h][w] = max(0.f, output_buf[h][w]);
    }
  }
//copy output here   
  copy_output:
  for (int h = 0; h < kOutImSize; ++h) {
    for (int w = 0; w < kOutImSize; ++w) {
      output[i*kOutImSize*kOutImSize + h*kOutImSize + w] = max(
          max(output_buf[h * 2][w * 2    ], output_buf[h * 2 + 1][w * 2    ]),
          max(output_buf[h * 2][w * 2 + 1], output_buf[h * 2 + 1][w * 2 + 1]));
    }
  }
}
}