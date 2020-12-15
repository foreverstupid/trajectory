#ifndef GPM_CUDA_INTEGRAL_GETTER_CLASS_HPP
#define GPM_CUDA_INTEGRAL_GETTER_CLASS_HPP

#include <cstdlib>
#include <cstdio>
#include "AbstractIntegralGetter.hpp"

/*
 * The wrapper of the kernel, that calculates correlation integral.
 */
class CudaIntegralGetter : public AbstractIntegralGetter
{
private:
    float *reductionOut;
    float *deviceInput;
    float *deviceReductionOut;

    int blockCount;
    int blockSize;

public:
    CudaIntegralGetter(const char *fileName, int N);
    ~CudaIntegralGetter();

    float getCorrelationIntegral(int k, float l);
};

#endif