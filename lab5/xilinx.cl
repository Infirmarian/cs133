__constant int kNum = 256;
__constant int kKernel = 5;
__constant int kImSize = 224;
__constant int kInImSize = 228;
__constant int kOutImSize = 112;

//#define input(j, h, w) \
//    input[((j) * kInImSize * kInImSize + (h) * kInImSize + (w))]

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void CnnKernel(__constant float* input, __constant float* weight,
               __constant float* bias, __global float* output) {

//  float output_buf[kImSize][kImSize] // buffer of output
//  ;
//  float input_buf[kInImSize][kInImSize + kKernel - 1][kKernel] //buffer of input
//  __attribute__((xcl_array_partition(cyclic, 8, 1)))  // cyclic partition factor of 8 in dim 1 of input_buf
//  __attribute__((xcl_array_partition(complete, 3))) // complete partitioning for dim3 of input_buf
//  ;

//  float weight_buf[kKernel][kKernel] //buffer of weight
//  __attribute__((xcl_array_partition(complete, 1))) // complete partitioning for dim 1 of weight_buf
//  __attribute__((xcl_array_partition(complete, 2))) // complete partitioning for dim 2 of weight_buf
//  ;

//copy bias here

//input load loop
//load_in:
//  for (int w = 0; w < kInImSize; w ++) {
//    __attribute__((xcl_pipeline_loop))
//    for (int q = 0; q < kKernel; ++q) { //make kKernel copy of input(j,h,w)
//      input_buf[h][w - q + kKernel - 1][q] = input(j, h, w);
//    }
//  }

//copy weight here

//convolution loop
//conv:
//  __attribute__((xcl_pipeline_loop))
//  for (int w = 0; w < kImSize; ++w) { //pipelined loop
//    float tmp = 0; 
//    for (int p = 0; p < kKernel; ++p) {  // unrolled loop
//      for (int q = 0; q < kKernel; ++q) {  //unrolled loop
//        tmp += //will be synthesized into tree reduction
//               weight_buf[p][q] *
//               input_buf[h + p][w + kKernel - 1][q];
//      }
//    }
//    output_buf[h][w] += tmp; //store reduction result
//  }

//copy output here          
}