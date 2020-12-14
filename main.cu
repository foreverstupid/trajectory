#include <stdio.h>

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



/*===================================================================*/
/*------------------------------ KERNELS ----------------------------*/
/*===================================================================*/
/*
 * Calculates values that are defined by the formula:
 *      \rho(i, j) = \sqrt(\sum_{s = 0}^{k - 1} (y_{j + s} - y_{i + s})^2).
 * Args:
 *      data:
 *          Input data as an array of 3-vectors.
 *
 *      idx1:
 *          The first correlating vector index.
 *
 *      idx2:
 *          The second correlating vector index.
 *
 *      k:
 *          The delay value.
 */
__device__
float getRho(float *data, int idx, int N, int k)
{
    int idx1 = idx / N;
    int idx2 = idx % N;

    float result = 0.0f;
    for (int i = 0; i < k; i++)
    {
        float x = data[3 * (idx1 + i)] - data[3 * (idx2 + i)];
        float y = data[3 * (idx1 + i) + 1] - data[3 * (idx2 + i) + 1];
        float z = data[3 * (idx1 + i) + 2] - data[3 * (idx2 + i) + 2];

        result += x * x + y * y + z * z;
    }

    return sqrt(result);
}



/*
 * Help sum reducing function. Accelerates the reducing via warps features.
 */
__device__
void warpReduce(volatile float *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}



/*
 * The Heaviside function.
 */
__device__
__forceinline__
float theta(float x)
{
    return x >= 0 ? 1.0f : 0.0f;
}



/*
 * Reduces the correlation integral. The outer code should sum all elements
 * of the output to get the integral value. This code performs the most
 * of the work. The integral is calculated usig the formula:
 *      C_k(l) = 1/N^2 \sum_{i, j} \theta(l - \rho(i, j)),
 * where \theta is the Heaviside function.
 * Args:
 *      rhos:
 *          Calculated by "calculateRhos" values of \rho(i, j).
 *
 *      output:
 *          The output of the function.
 *
 *      N:
 *          Data series elements count.
 *
 *      l:
 *          The "l" value for the integral.
 */
__global__
void reduceCorrelationIntegral(
    float *input,
    float *output,
    int N,
    int k,
    float l)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (2 * blockDim.x) + tid;

    sdata[tid] =
        theta(l - getRho(input, i, N, k)) +
        theta(l - getRho(input, i + blockDim.x, N, k));

    __syncthreads();

    for (int j = blockDim.x / 2; j > 32; j >>=1)
    {
        if (tid < j)
        {
            sdata[tid] += sdata[tid + j];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        warpReduce(sdata, tid);
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}



/*=================================================================*/
/*---------------------------- HOST CODE --------------------------*/
/*=================================================================*/
int getData(const char *inputFileName, int maxDataCount, float *data)
{
    FILE *file = fopen(inputFileName, "rb");
    int readElements = fread((void *)data, sizeof(float), 3 * maxDataCount, file);
    fclose(file);
    return readElements;
}



int main(int argc, char **argv)
{
    int N;
    float origin;
    float step;

    sscanf(argv[2], "%d", &N);
    sscanf(argv[3], "%f", &origin);
    sscanf(argv[4], "%f", &step);

    int dataSize = N + 10;  // additional elements for k-shifting
    int blockSize = 512;
    int blockCount = (N * N + blockSize - 1) / blockSize / 2;

    float *input = new float[3 * dataSize];
    float *reductionOut = new float[blockCount];

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

    float *deviceReductionOut;
    float *deviceInput;

    _( cudaMalloc(&deviceReductionOut, blockCount * sizeof(float)) );
    _( cudaMalloc(&deviceInput, 3 * dataSize * sizeof(float)) );

    _( cudaMemcpy(
        deviceInput,
        input,
        3 * dataSize * sizeof(float),
        cudaMemcpyHostToDevice) );

    for (int k = 1; k < 7; k++)
    {
        char name[120];
        sprintf(name, "plot_%d.ssv", k);
        FILE *out = fopen(name, "w");

        float l = origin;
        for (int q = 0; q < 50; q++)
        {
            reduceCorrelationIntegral
                <<<blockCount, blockSize, blockSize * sizeof(float)>>>
                (deviceInput, deviceReductionOut, N, k, l);

            _( cudaMemcpy(
                reductionOut,
                deviceReductionOut,
                blockCount * sizeof(float),
                cudaMemcpyDeviceToHost) );

            float integral = 0.0;
            for (int t = 0; t < blockCount; t++)
            {
                integral += reductionOut[t];
            }

            integral /= N * N;
            fprintf(out, "%e %e\n", log(l), log(integral));
            l *= step;
        }

        fclose(out);
    }

    delete[] input;
    delete[] reductionOut;

    cudaFree(deviceInput);
    cudaFree(deviceReductionOut);

    return 0;
}