#ifndef TIMING_HPP
#define TIMING_HPP
#include <string>

namespace TimingModule {
    // Timer controls
    void init_timer();
    
    void start_math_timer();
    void stop_math_timer();
    
    void start_io_timer();
    void stop_io_timer();
    
    // Output generator
    void print_timing_results(int nx, int ny, const std::string& arch);
}

#endif // TIMING_HPP