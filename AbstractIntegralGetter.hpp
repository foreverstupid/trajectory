#ifndef GPM_ABSTRACT_INTEGRAL_GETTER_CLASS_HPP
#define GPM_ABSTRACT_INTEGRAL_GETTER_CLASS_HPP

#include <cstdlib>
#include <cstdio>

class AbstractIntegralGetter
{
protected:
    float *input;
    int N;
    int dataSize;   // expanded N for k-shifting

public:
    AbstractIntegralGetter(const char *fileName, int N);
    virtual ~AbstractIntegralGetter();

    virtual float getCorrelationIntegral(int k, float l) = 0;
};

#endif