CXX = mpicxx

SRCS = ../lab1/gemm.h ../lab1/gemm.cpp ../lab1/gemm-baseline.a \
       mpi.cpp main.cpp

REQUIRED_FILES = mpi.cpp lab2-report.pdf

np ?= 4

test: gemm
	mpiexec -np $(np) ./$^

gemm: $(SRCS)
	$(CXX) $(subst -fopenmp,,$(CXXFLAGS)) -o $@ $(filter %.cpp %.a %.o, $^) -lgomp

../lab1/gemm-baseline.a:
	$(MAKE) -C ../lab1 gemm-baseline.a

include ../makefile
