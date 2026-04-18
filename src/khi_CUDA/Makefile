# Compiler settings
NVCC = nvcc
INCLUDES = -I../../include/common
# -O3 for maximum optimization, -use_fast_math is the CUDA equivalent of -ffast-math
NVCCFLAGS = -O3 -use_fast_math -Xcompiler "-Wall -Wextra" $(INCLUDES)

# Target executable
TARGET = cuda_khi

# Source files
CU_SRCS = main.cu
CPP_SRCS = ../common/timing.cpp
HEADERS = ../../include/common/timing.hpp	

.PHONY: all compile run clean

# Default rule triggers both compilation and execution
all: compile run

compile: $(TARGET)

$(TARGET): $(CU_SRCS) $(CPP_SRCS) $(HEADERS)
	@echo "Compiling CUDA KHI solver..."
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(CU_SRCS) $(CPP_SRCS)
	@echo "Compilation successful! Executable '$(TARGET)' created."

run: $(TARGET)
	@echo "Preparing centralized results directory..."
	@mkdir -p ../../results/CUDA_VTK
	@rm -f ../../results/CUDA_VTK/*.vtk
	@echo "Starting CUDA simulation..."
	@# Execute the program from INSIDE the top-level results folder
	@# This ensures benchmarks.csv is created in the shared results/ folder, not locally
	@cd ../../results && ../src/khi_CUDA/$(TARGET)
	@# Move the VTK files cleanly into their own subfolder so they don't clutter the root results folder
	@mv ../../results/*.vtk ../../results/CUDA_VTK/ 2>/dev/null || true
	@echo "---------------------------------------------------"
	@echo "Simulation finished! Shared benchmarks.csv updated in the 'results/' folder."

clean:
	@echo "Cleaning executables..."
	rm -f $(TARGET)