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
float getRho(float *data, int idx1, int idx2, int k)
{
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
 * Calculates the matrix [ \rho(i, j) ] for all i, j in 1..N.
 * Args:
 *      input:
 *          3D data series contains N 3-vectors as a plain array.
 *
 *      output:
 *          Placeholder for N^2 float values that will contain \rho(i, j)
 *          in a row major format.
 *
 *      N:
 *          Number of input data vectors.
 *
 *      k:
 *          The delay value.
 */
__global__
void calculateRhos(
    float *input,
    float *output,
    int N,
    int k)
{
    extern __shared__ float shared[];

    int idx1 = threadIdx.x;
    int idx2 = threadIdx.y;

    int m = blockDim.x;
    int n = blockDim.y;

    int globalIdx1 = m * blockIdx.x + idx1;
    int globalIdx2 = n * blockIdx.y + idx2;

    if (idx2 == 0)
    {
        shared[idx1 * 3] = input[globalIdx1 * 3];
        shared[idx1 * 3 + 1] = input[globalIdx1 * 3 + 1];
        shared[idx1 * 3 + 2] = input[globalIdx1 * 3 + 2];
    }
    else if (idx2 == 1 && idx1 < k)
    {
        shared[(m + idx1) * 3] = input[(globalIdx1 + idx1) * 3];
        shared[(m + idx1) * 3 + 1] = input[(globalIdx1 + idx1) * 3 + 1];
        shared[(m + idx1) * 3 + 2] = input[(globalIdx1 + idx1) * 3 + 2];
    }

    __syncthreads();

    float rho = getRho(shared, idx1, idx2, k);
    output[globalIdx1 * N + globalIdx2] = rho;
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
    return x > 0 ? x : 0.0f;
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
    float *rhos,
    float *output,
    int N,
    float l)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (2 * blockDim.x) + tid;

    float coeff = 1.0f / (N * N);
    sdata[tid] = 
        coeff * (theta(l - rhos[i]) + theta(l - rhos[i + blockDim.x]));
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
int main(int argc, char **argv)
{
    int dataSize = 10000;
    int blockSize = 256;
    int blockCount = (dataSize + blockSize - 1) / blockSize;

    dim3 gridSize2D = dim3(blockCount, blockCount);
    dim3 blockSize2D = dim3(blockSize, blockSize);

    int N = 500;
    int k = 5;
    float l = 1e-7;
    int blockCountReduction = (N * N + blockSize - 1) / blockSize / 2;

    float *input = new float[3 * dataSize];
    float *rhos = new float[dataSize * dataSize];
    float *reductionOut = new float[blockSize];

    float *deviceInput;
    float *deviceRhos;
    float *deviceReductionOut;

    _( cudaMalloc(&deviceInput, 3 * dataSize * sizeof(float)) );
    _( cudaMalloc(&deviceRhos, dataSize * dataSize * sizeof(float)) );
    _( cudaMalloc(&deviceReductionOut, blockSize * sizeof(float)) );

    for (int i = 0; i < dataSize; i++)
    {
        input[i * 3] = sin(i);
        input[i * 3 + 1] = cos(i);
        input[i * 3 + 2] = i;
    }

    float elapsed = 0.0;
    cudaEvent_t start;
    cudaEvent_t stop;
    _( cudaEventCreate(&start) );
    _( cudaEventCreate(&stop) );
    _( cudaMemcpy(
            deviceInput,
            input,
            3 * dataSize * sizeof(float),
            cudaMemcpyHostToDevice) );

    _( cudaEventRecord(start, 0) );
    calculateRhos
        <<<gridSize2D, blockSize2D, 3 * (blockSize + k) * sizeof(float)>>>
        (deviceInput, deviceRhos, N, k);

    reduceCorrelationIntegral
        <<<blockCountReduction, blockSize, blockSize * sizeof(float)>>>
        (deviceRhos, deviceReductionOut, N, l);
    _( cudaEventRecord(stop, 0) );

    _( cudaEventSynchronize(stop) );
    _( cudaEventElapsedTime(&elapsed, start, stop) );
    _( cudaEventDestroy(start) );
    _( cudaEventDestroy(stop) );

    _( cudaMemcpy(
            reductionOut,
            deviceReductionOut,
            blockSize * sizeof(float),
            cudaMemcpyDeviceToHost) );

    float integral = 0.0;
    for (int i = 0; i < blockSize; i++)
    {
        integral += reductionOut[i];
    }

    printf("%e -> %e\n", integral, log(integral) / log(l));
    printf("GPU time: %e ms\n", elapsed);

    _( cudaFree(deviceInput) );
    _( cudaFree(deviceRhos) );
    _( cudaFree(deviceReductionOut) );
    delete[] input;
    delete[] rhos;
    delete[] reductionOut;

    return 0;
}