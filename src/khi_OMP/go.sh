#!/bin/bash

ulimit -s unlimited     # prevent overflow in large simulations
export OMP_STACKSIZE=1G     # Safety Margin of 1GB thread stack size to prevent overflow
export OMP_NUM_THREADS=20

echo "Preparing centralized results directory..."
mkdir -p ../../results/OMP_VTK
rm -f ../../results/OMP_VTK/*.vtk

echo "Starting OpenMP simulation..."
cd ../../results
time ../src/khi_OMP/omp_khi

mv *.vtk OMP_VTK/ 2>/dev/null || true

echo "---------------------------------------------------"
echo "Simulation finished! View the results in ParaView."