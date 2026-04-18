#!/bin/bash

echo "Compiling OpenMP KHI solver..."

# Added -fopenmp to enable threading
gfortran -O3 -ffast-math -fopenmp -o omp_khi ../common/timing.f90 main.f90
