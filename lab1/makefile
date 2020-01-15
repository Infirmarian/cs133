SRCS = gemm.h gemm.cpp main.cpp omp.cpp omp-blocked.cpp gemm-baseline.a

REQUIRED_FILES = omp.cpp omp-blocked.cpp lab1-report.pdf

test: gemm
	./$^ parallel parallel-blocked

gemm: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a, $^)

include ../makefile
