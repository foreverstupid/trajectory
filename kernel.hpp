#ifndef GPM_KERNEL_CUDA_MODULE_HPP
#define GPM_KERNEL_CUDA_MODULE_HPP

#ifdef CUDA
/*
 * The wrapper of the kernel, that calculates correlation integral.
 */
float getCorrelationIntegral(
    float *deviceInput,
    float *deviceReductionOut,
    flaot *reductionOut,
    int N,
    int k,
    float l,
    int blockCount,
    int blockSize);
#endif

#endif