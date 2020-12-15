NVCC=nvcc
MPICC=mpicxx
CC=g++
CFLAGS=-O2

cudampi: main.cpp AbstractIntegralGetter.o CudaIntegralGetter.o
	$(MPICC) $(CFLAGS) -DCUDA -DMPI -L/usr/local/cuda/lib64 $^ -lm -lcudart -o gpm_mpicuda

cuda: main.cpp AbstractIntegralGetter.o CudaIntegralGetter.o
	$(NVCC) $(CFLAGS) -DCUDA $^ -lm -o gpm_cuda

mpi: main.cpp AbstractIntegralGetter.o IntegralGetter.o
	$(MPICC) $(CFLAGS) -DMPI $^ -lm -o gpm_mpi

simple: main.cpp AbstractIntegralGetter.o IntegralGetter.o
	$(CC) $(CFLAGS) $^ -lm -o gpm_simple

AbstractIntegralGetter.o: AbstractIntegralGetter.cpp AbstractIntegralGetter.hpp
	$(CC) $(CFLAGS) -c $< -o $@

IntegralGetter.o: IntegralGetter.cpp IntegralGetter.hpp
	$(CC) $(CFLAGS) -c $< -o $@

CudaIntegralGetter.o: CudaIntegralGetter.cu CudaIntegralGetter.hpp
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o gpm*
