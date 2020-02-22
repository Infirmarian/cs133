SRCS = cnn.h cnn.cpp opencl.cpp main.cpp

REQUIRED_FILES = lab4-report.pdf nvidia.cl params.sh

export OPENCL_PLATFORM= NVIDIA CUDA
export OPENCL_DEVICE= Tesla M60
export OPENCL_SOURCE ?= nvidia.cl

test: cnn $(OPENCL_SOURCE)
	. ./params.sh; ./$<

test-seq:
	@$(MAKE) --no-print-directory test SEQUENTIAL=

test-vadd: vadd vadd.cl
	OPENCL_SOURCE=vadd.cl ./$<

cnn: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

vadd: opencl.cpp vadd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.cpp %.a %.o, $^) $(LDFLAGS)

include ../makefile

CXXFLAGS := $(subst -march=native,,$(CXXFLAGS))
LDFLAGS += -lOpenCL
