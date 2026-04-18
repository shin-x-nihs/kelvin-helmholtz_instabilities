# Compiler settings
CXX = g++
INCLUDES = -I../../include/common
CXXFLAGS = -O3 -Wall -Wextra -Werror -flto -ffast-math -Wdouble-promotion $(INCLUDES)

# Target executable
TARGET = serial_cpp

# Source files
SRCS = main.cpp
SRCS += ../common/timing.cpp
HEADERS = ../../include/common/timing.hpp

.PHONY: all compile run clean

# Default rule triggers both compilation and execution
all: compile run

compile: $(TARGET)

$(TARGET): $(CU_SRCS) $(CPP_SRCS) $(HEADERS)
	@echo "Compiling Pure Serial C++ KHI solver..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
	@echo "Compilation successful! Executable '$(TARGET)' created."

run: $(TARGET)
	@echo "Preparing centralized results directory..."
	@mkdir -p ../../results/C++_VTK
	@rm -f ../../results/C++_VTK/*.vtk
	@echo "Starting Pure Serial C++ simulation..."
	@# Execute the program from INSIDE the top-level results folder
	@# This ensures benchmarks.csv is created in the shared results/ folder, not locally
	@cd ../../results && ../src/khi_BASELINE_C++/$(TARGET)
	@# Move the VTK files cleanly into their own subfolder so they don't clutter the root results folder
	@mv ../../results/*.vtk ../../results/C++_VTK/ 2>/dev/null || true
	@echo "---------------------------------------------------"
	@echo "Simulation finished! Shared benchmarks.csv updated in the 'results/' folder."

clean:
	@echo "Cleaning executables..."
	rm -f $(TARGET)