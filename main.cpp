#include <cstdio>
#include <ctime>
#include <cmath>
#include "AbstractIntegralGetter.hpp"

#ifdef MPI
#include <mpi.h>
#endif

#ifdef CUDA
#include "CudaIntegralGetter.hpp"
#else
#include "IntegralGetter.hpp"
#endif



#ifdef MPI
/*
 * Initializes MPI, returning the rank of the process.
 */
int initMPI(int argc, char **argv)
{
    int errCode;
    if ((errCode = MPI_Init(&argc, &argv)) != 0)
    {
        fprintf(stderr, "!Cannot init MPI: error code is %d\n", errCode);
        exit(1);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}
#endif



/*
 * Estimates and prints the running time of the program.
 */
void estimateRunningTime(clock_t start)
{
#ifdef MPI
    float local_seconds = (float)(clock() - start) / CLOCKS_PER_SEC;
    float seconds;
    MPI_Reduce(
        &local_seconds, &seconds,
        1, MPI_FLOAT, MPI_MAX,
        0, MPI_COMM_WORLD);
#else
    float seconds = (float)(clock() - start) / CLOCKS_PER_SEC;
#endif

    printf("Running time: %f seconds\n", seconds);
}



int main(int argc, char **argv)
{
    clock_t start = clock();

#ifdef MPI
    int rank = initMPI(argc, argv);
#endif

    int N;
    float origin;
    float step;

    sscanf(argv[2], "%d", &N);
    sscanf(argv[3], "%f", &origin);
    sscanf(argv[4], "%f", &step);

#ifdef CUDA
    CudaIntegralGetter ig = CudaIntegralGetter(argv[1], N);
#else
    IntegralGetter ig = IntegralGetter(argv[1], N);
#endif

#ifdef MPI
    int k = rank + 1;
#else
    for (int k = 1; k < 7; k++)
#endif
    {
        char name[120];
        sprintf(name, "plot_%d.ssv", k);
        FILE *out = fopen(name, "w");

        float l = origin;
        for (int q = 0; q < 50; q++)
        {
            float integral = ig.getCorrelationIntegral(k, l);
            fprintf(out, "%e %e\n", log(l), log(integral));
            l *= step;
        }

        fclose(out);
    }

    estimateRunningTime(start);

#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}