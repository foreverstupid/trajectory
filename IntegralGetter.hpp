#ifndef GPM_INTEGRAL_GETTER_CLASS_HPP
#define GPM_INTEGRAL_GETTER_CLASS_HPP

#include <cmath>
#include "AbstractIntegralGetter.hpp"

class IntegralGetter : public AbstractIntegralGetter
{
public:
    IntegralGetter(const char *fileName, int N)
        : AbstractIntegralGetter(fileName, N)
    {
    }

    float getCorrelationIntegral(int k, float l);
};

#endif