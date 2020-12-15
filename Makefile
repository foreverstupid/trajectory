mpicuda: main.cpp kernel.cu kernel.hpp
	nvcc -O2 -DMPI -DCUDA -c kernel.cu -o kernel.o
	mpicxx -DMPI -DCUDA -c main.cpp -o main.o
	mpicxx main.o kernel.o -lcudart -o gpm-mc

cuda: main.cpp kernel.cu kernel.hpp
	nvcc -O2 -DCUDA -c kernel.cu -o kernel.o
	nvcc -DCUDA -c main.cpp -o main.o
	nvcc main.o kernel.o -o gpm-c

mpi: main.cpp
	mpicxx -O2 -DMPI -o gpm-m main.cpp

simple: main.cpp
	g++ -O2 -o gpm main.cpp

clean:
	rm -f *.o gpm*