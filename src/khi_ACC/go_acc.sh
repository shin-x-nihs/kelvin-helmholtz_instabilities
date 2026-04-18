#!/bin/bash

export NVCOMPILER_ACC_TIME=1

echo "Preparing centralized results directory..."
mkdir -p ../../results/ACC_VTK
rm -f ../../results/ACC_VTK/*.vtk

echo "Starting OpenACC simulation..."
cd ../../results
time ../src/khi_ACC/acc_khi

mv *.vtk ACC_VTK/ 2>/dev/null || true

echo "---------------------------------------------------"
echo "Simulation finished! View the results in ParaView."