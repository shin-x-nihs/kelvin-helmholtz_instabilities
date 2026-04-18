#!/bin/bash

echo "Compiling Serial KHI solver..."

# Added -fopenmp to enable threading
gfortran -O3 -ffast-math -o serial_khi ../common/timing.f90 main.f90

if [ $? -eq 0 ]; then
    echo "Compilation successful! Executable 'serial_khi' created."
else
    echo "Compilation failed."
fi