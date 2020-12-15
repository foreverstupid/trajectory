#include "IntegralGetter.hpp"

/*
 * Heaviside theta-function.
 */
static inline float theta(float x)
{
    return x >= 0 ? 1.0 : 0.0;
}



/*
 * Squares the given argument.
 */
static inline float sqr(float x)
{
    return x * x;
}



float IntegralGetter::getCorrelationIntegral(int k, float l)
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