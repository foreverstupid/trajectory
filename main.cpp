#include <stdio.h>
#include <time.h>

#ifdef MPI
#include <mpi.h>
#endif

#ifdef CUDA
#include "kernel.hpp"
#define _(code) do{ gpuAssert(code, __FILE__, __LINE__); }while(0)

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(
            stderr, "![%s:%d] %s: %s\n",
            file, line, cudaGetErrorName(code), cudaGetErrorString(code));

        exit(1);
    }
}

#else

inline float theta(float x)
{
    return x >= 0 ? 1.0 : 0.0;
}



inline float sqr(float x)
{
    return x * x;
}



float getCorrelationIntegral(const float *input, float l, int k, int N)
{
    float result = 0.0;
    for (int i = 0; i < N; i++)
    {
        for (int j = i + 1; j < N; j++)
        {
            float rho = 0.0;
            for (int p = 0; p < k; p++)
            {
                rho +=
                    sqr(input[(i + p) * 3] - input[(j + p) * 3]) +
                    sqr(input[(i + p) * 3 + 1] - input[(j + p) * 3 + 1]) +
                    sqr(input[(i + p) * 3 + 2] - input[(j + p) * 3 + 2]);
            }

            result += sqrt(rho);
        }
    }

    return result / (N  * N);
}
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

    printf("Running time: %f seconds", seconds);
}



/*
 * Reads the input data from the given file. Each piece of the data is
 * a 3-vector of float. This function returns actually read floats count.
 */
int getData(const char *inputFileName, int maxDataCount, float *data)
{
    FILE *file = fopen(inputFileName, "rb");
    int readElements = fread((void *)data, sizeof(float), 3 * maxDataCount, file);
    fclose(file);
    return readElements;
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

    int dataSize = N + 10;  // additional elements for k-shifting
    float *input = new float[3 * dataSize];

#ifdef CUDA
    int blockSize = 512;
    int blockCount = (N * N + blockSize - 1) / blockSize / 2;
    float *reductionOut = new float[blockCount];
#endif

    int actualDataSize = getData(argv[1], dataSize, input);
    if (getData(argv[1], dataSize, input) != 3 * dataSize)
    {
        fprintf(
            stderr,
            "!File contains only %d elements (%d required)\n",
            actualDataSize,
            3 * dataSize);

        exit(1);
    }

#ifdef CUDA
    float *deviceReductionOut;
    float *deviceInput;

    _( cudaMalloc(&deviceReductionOut, blockCount * sizeof(float)) );
    _( cudaMalloc(&deviceInput, 3 * dataSize * sizeof(float)) );

    _( cudaMemcpy(
        deviceInput,
        input,
        3 * dataSize * sizeof(float),
        cudaMemcpyHostToDevice) );
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
            float integral = 0.0;

#ifdef CUDA
            integral = getCorrelationIntegral(
                deviceInput, deviceReductionOut, reductionOut,
                N, k, l,
                blockCount, blockSize);
#else
            integral = getCorrelationIntegral(input, l, k, N);
#endif
            fprintf(out, "%e %e\n", log(l), log(integral));
            l *= step;
        }

        fclose(out);
    }

    delete[] input;

#ifdef CUDA
    delete[] reductionOut;
    cudaFree(deviceInput);
    cudaFree(deviceReductionOut);
#endif

    estimateRunningTime(start);

#ifdef MPI
    MPI_Finalize();
#endif

    return 0;
}