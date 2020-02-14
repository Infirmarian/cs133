SRCS = cnn.h cnn.cpp opencl.cpp main.cpp

REQUIRED_FILES = lab3-report.pdf intel.cl params.sh

export OPENCL_DEVICE ?= Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz
export OPENCL_PLATFORM ?= Intel(R) CPU Runtime for OpenCL(TM) Applications
export OPENCL_SOURCE ?= intel.cl

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
