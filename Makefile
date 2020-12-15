cudampi: main.cpp AbstractIntegralGetter.o CudaIntegralGetter.o
	mpicxx -O2 -DCUDA -DMPI $^ -lm -o gpm_mpicuda

cuda: main.cpp AbstractIntegralGetter.o CudaIntegralGetter.o
	nvcc -O2 -DCUDA $^ -lm -o gpm_cuda

mpi: main.cpp AbstractIntegralGetter.o IntegralGetter.o
	mpicxx -O2 -DMPI $^ -lm -o gpm_mpi

simple: main.cpp AbstractIntegralGetter.o IntegralGetter.o
	g++ -O2 $^ -lm -o gpm_simple

AbstractIntegralGetter.o: AbstractIntegralGetter.cpp AbstractIntegralGetter.hpp
	g++ -O2 -c $< -o $@

IntegralGetter.o: IntegralGetter.cpp IntegralGetter.hpp
	g++ -O2 -c $< -o $@

CudaIntegralGetter.o: CudaIntegralGetter.cu CudaIntegralGetter.hpp
	nvcc -O2 -c $< -o $@

clean:
	rm -f *.o gpm*