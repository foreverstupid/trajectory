#include "AbstractIntegralGetter.hpp"

/*
 * Reads the input data from the given file. Each piece of the data is
 * a 3-vector of float. This function returns actually read floats count.
 */
static int getData(const char *inputFileName, int maxDataCount, float *data)
{
    FILE *file = fopen(inputFileName, "rb");
    int readElements = fread((void *)data, sizeof(float), 3 * maxDataCount, file);
    fclose(file);
    return readElements;
}



AbstractIntegralGetter::AbstractIntegralGetter(
    const char *fileName,
    int N)
{
    this->N = N;
    dataSize = N + 10;  // additional elements for k-shifting

    float *input = new float[3 * this->dataSize];

    int actualDataSize = getData(fileName, dataSize, input);
    if (getData(fileName, dataSize, input) != 3 * dataSize)
    {
        fprintf(
            stderr,
            "!File contains only %d elements (%d required)\n",
            actualDataSize,
            3 * dataSize);

        exit(1);
    }
}



AbstractIntegralGetter::~AbstractIntegralGetter()
{
    delete[] input;
}