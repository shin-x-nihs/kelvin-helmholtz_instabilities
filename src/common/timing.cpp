#include "timing.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>  
#include <string>  

namespace TimingModule {
    using namespace std::chrono;
    
    // Internal timing variables (hidden from main.cpp)
    static high_resolution_clock::time_point start_math;
    static high_resolution_clock::time_point end_math;
    static high_resolution_clock::time_point start_io;
    static high_resolution_clock::time_point end_io;

    static double total_math_time = 0.0;
    static double total_io_time = 0.0;
    static long long total_steps = 0;

    void init_timer() {
        total_math_time = 0.0;
        total_io_time = 0.0;
        total_steps = 0;
        std::cout << "Hardware Timer Initialized (std::chrono)." << std::endl;
    }

    void start_math_timer() {
        start_math = high_resolution_clock::now();
    }

    void stop_math_timer() {
        end_math = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end_math - start_math);
        total_math_time += time_span.count();
        total_steps++;
    }

    void start_io_timer() {
        start_io = high_resolution_clock::now();
    }

    void stop_io_timer() {
        end_io = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(end_io - start_io);
        total_io_time += time_span.count();
    }

    void print_timing_results(int nx, int ny, const std::string& arch) {
        double mcups = 0.0;
        if (total_math_time > 0.0) {
            // Calculate MCUPS: (nx * ny * total_steps) / (math_time * 10^6)
            mcups = (static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(total_steps)) / (total_math_time * 1.0e6);
        }

        // --- 1. Print to Standard Terminal ---
        std::cout << "------------------------------------------------\n";
        std::cout << "               BENCHMARK RESULTS                \n";
        std::cout << "------------------------------------------------\n";
        std::cout << "Architecture:      " << arch << "\n";
        std::cout << "Resolution:        " << nx << "x" << ny << "\n";
        std::cout << "Total Steps:       " << total_steps << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Math Time (Compute): " << total_math_time << " seconds\n";
        std::cout << "I/O Time (Disk/IO):  " << total_io_time << " seconds\n";
        std::cout << "Total Wall-Clock:  " << (total_math_time + total_io_time) << " seconds\n";
        std::cout << std::setprecision(4);
        std::cout << "Performance:       " << mcups << " MCUPS\n";
        std::cout << "------------------------------------------------\n";

        // --- 2. Append to CSV for Automation ---
        std::string filename = "benchmarks.csv";
        
        // Check if the file already exists
        std::ifstream file_check(filename);
        bool file_exists = file_check.good();
        file_check.close();

        // Open file in append mode
        std::ofstream csv_file(filename, std::ios::app);
        if (csv_file.is_open()) {
            // If the file is brand new, write the CSV headers first
            if (!file_exists) {
                csv_file << "Language,Architecture,GridSize,MathTime_sec,IOTime_sec,MCUPS\n";
            }

            // Write the data row with strict formatting
            csv_file << std::fixed << std::setprecision(6);
            csv_file << "C++," << arch << "," << nx << "," 
                     << total_math_time << "," << total_io_time << "," 
                     << mcups << "\n";
                     
            csv_file.close();
        } else {
            std::cerr << "Warning: Could not open " << filename << " to write benchmark data.\n";
        }
    }
}