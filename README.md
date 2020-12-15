# Grassberger-Procaccia Method

This prigram is crated as a task for the course "Supercomputer modeling and techonologies". It uses Grassberger-Procaccia method for determining the embedding dimension of some time series.

## Build

There are four versions of the code. One that implements a sequential algorithm, one that uses MPI for paralleling calculation for different dimensions, one that uses Cuda for acceleration sum reduction (but without additional dimension paralleling), and one that uses both MPI and Cuda. For building each version use the follownig `make` invocations:

```
make simple     # for sequential version
make mpi        # for MPI version
make cuda       # for Cuda version
make cudampi    # for Cuda + MPI version
```

## Run

The program takes the following arguments:

- Data file path. It should be a binary file, that contains a set of 3-vectors as a float numbers collection.
- Point count. It defines how many 3-vectors from the data file will be considered for calculations.
- Start point for l. The l values range start.
- l step. The step for l values range in log scale.
- Output data prefix. All output files will have this prefix.

## Data collection

The program creates a collection of files (one file for each dimesion), that are space separated text files contains l values and correlation integrals for such l values. After that you can use a special Python3 script for plotting data:

```
./plot.py <prefix> <dim>
```
where `<prefix>` is specified prefix for the algorithm and `<dim>` is dimensions number. This script will generate PNG file `<prefix>.png` with plotted data.