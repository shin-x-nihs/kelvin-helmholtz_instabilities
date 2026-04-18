#!/bin/bash

echo "Lifting stack memory limits for OpenMP..."
ulimit -s unlimited         # prevent overflow in large simulations

echo "Preparing centralized results directory..."
mkdir -p ../../results/FORTRAN_VTK
rm -f ../../results/FORTRAN_VTK/*.vtk

echo "Starting Fortran simulation..."
cd ../../results
time ../src/khi_BASELINE/serial_khi

mv *.vtk FORTRAN_VTK/ 2>/dev/null || true

echo "---------------------------------------------------"
echo "Simulation finished! View the results in ParaView."