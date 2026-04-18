#!/bin/bash

echo "--- Compiling for NVIDIA GPU ---"
# Compiler Flags Breakdown:
# -O3         : Maximum CPU optimization for any serial parts of the code
# -acc        : Enables OpenACC directives
# -gpu=cc90   : Targets the Blackwell architecture (RTX 50-series)
# -Minfo=accel: Prints exact details on which loops were successfully parallelized

nvfortran -O3 -acc -gpu=ccall -Minfo=accel ../common/timing.f90 main.f90  -o acc_khi
