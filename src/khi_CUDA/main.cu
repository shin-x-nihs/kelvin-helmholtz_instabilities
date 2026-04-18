#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include "timing.hpp"

// --- Simulation Parameters ---
const int nx = 512;
const int ny = 512;
const int ng = 3; 
const int nx_tot = nx + 2 * ng;
const int ny_tot = ny + 2 * ng;
const float Lx = 1.0f;
const float Ly = 1.0f;
const float dx = Lx / nx;
const float dy = Ly / ny;
const float inv_dx = 1.0f / dx;
const float inv_dy = 1.0f / dy;
const float CFL_target = 0.4f;
const float t_max = 2.0f;
const float gamma_air = 1.4f;
const float output_interval = 0.05f;

// --- CUDA Error Checking Macro --- Error Handling is explicitly done after every CUDA call to ensure robustness and easier debugging.
// __FILE__ and __LINE__ are predifined macros built in compiler
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// __host__: This function can be called by the CPU.
// __device__: This function can be called by the GPU.    
// --- Flat 1D Array Indexing (Structure of Arrays layout) --- 
__host__ __device__ inline int idx(int c, int i, int j) {
    return (i + j * nx_tot) + c * (nx_tot * ny_tot);
}

// --- Endian Swapping for Binary VTK (Host Only) ---
uint32_t swap_endian(uint32_t val) {
    return (val << 24) | ((val << 8) & 0x00FF0000) | ((val >> 8) & 0x0000FF00) | (val >> 24);
}
float swap_float(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    i = swap_endian(i);
    float r;
    std::memcpy(&r, &i, sizeof(float));
    return r;
}

// --- Device Atomic Max for Single Precision Float --- read float as int --- atomicCAS(address, compare_value, new_value)
__device__ static float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed)))); // CUDA intrinsic functions to reinterpret bits between float and int without changing the bit pattern
    } while (assumed != old);
    return __int_as_float(old);
}

// --- WENO5 Reconstruction (Fast Math, Device Only) ---
__device__ inline float weno5(float v1, float v2, float v3, float v4, float v5) {
    const float eps = 1.0e-6f;
    
    float diff1 = v1 - 2.0f*v2 + v3;
    float diff2 = v1 - 4.0f*v2 + 3.0f*v3;
    float b0 = (13.0f/12.0f) * (diff1 * diff1) + 0.25f * (diff2 * diff2);

    float diff3 = v2 - 2.0f*v3 + v4;
    float diff4 = v2 - v4;
    float b1 = (13.0f/12.0f) * (diff3 * diff3) + 0.25f * (diff4 * diff4);

    float diff5 = v3 - 2.0f*v4 + v5;
    float diff6 = 3.0f*v3 - 4.0f*v4 + v5;
    float b2 = (13.0f/12.0f) * (diff5 * diff5) + 0.25f * (diff6 * diff6);

    float d0 = eps + b0;
    float d1 = eps + b1;
    float d2 = eps + b2;
    
    float a0 = 0.1f / (d0 * d0);
    float a1 = 0.6f / (d1 * d1);
    float a2 = 0.3f / (d2 * d2);
    
    float sum_a = a0 + a1 + a2;

    float w0 = a0 / sum_a;
    float w1 = a1 / sum_a;
    float w2 = a2 / sum_a;

    return w0 * ((1.0f/3.0f)*v1 - (7.0f/6.0f)*v2 + (11.0f/6.0f)*v3) +
           w1 * (-(1.0f/6.0f)*v2 + (5.0f/6.0f)*v3 + (1.0f/3.0f)*v4) +
           w2 * ((1.0f/3.0f)*v3 + (5.0f/6.0f)*v4 - (1.0f/6.0f)*v5);
}

// =========================================================================
//                             CUDA KERNELS
// =========================================================================

// Completely parallelized X boundaries mapping (2D Threads: j and k)
__global__ void apply_bcs_x_kernel(float* s) {
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Ghost cell depth // blockIdx.x=0 since gridDim.x=1 (gridDim.y=32) --> so k goes from 0 to 2 (inner loop in C++)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Y axis

    if (k < ng && j < ny_tot) {
        for (int c = 0; c < 4; ++c) {
            s[idx(c, k, j)] = s[idx(c, nx + k, j)];
            s[idx(c, nx + ng + k, j)] = s[idx(c, ng + k, j)];
        }
    }
}

// Completely parallelized Y boundaries mapping (2D Threads: i and k)
__global__ void apply_bcs_y_kernel(float* s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // X axis
    int k = blockIdx.y * blockDim.y + threadIdx.y; // Ghost cell depth

    if (i < nx_tot && k < ng) {
        for (int c = 0; c < 4; ++c) {
            s[idx(c, i, k)] = s[idx(c, i, ny + k)];
            s[idx(c, i, ny + ng + k)] = s[idx(c, i, ng + k)];
        }
    }
}

// Shared Memory Parallel Reduction to fix the atomic bottleneck
__global__ void compute_dt_kernel(const float* s, float* max_speed) {
    extern __shared__ float sdata[]; // Dynamic shared memory
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 1D thread ID within the block

    float local_max = 0.0f;
    
    // 1. Every thread computes its own cell's speed
    if (i >= ng && i < nx + ng && j >= ng && j < ny + ng) {
        float rho = s[idx(0, i, j)];
        float u   = s[idx(1, i, j)] / rho;
        float v   = s[idx(2, i, j)] / rho;
        float P   = (gamma_air - 1.0f) * (s[idx(3, i, j)] - 0.5f * rho * (u*u + v*v));
        float c   = sqrtf(gamma_air * P / rho);
        local_max = fmaxf(fabsf(u) + c, fabsf(v) + c);
    }
    
    // 2. Load into ultra-fast block shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // 3. Tree-based parallel reduction
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // 4. Thread 0 writes the block's maximum to the global atomic lock
    if (tid == 0) {
        atomicMaxFloat(max_speed, sdata[0]);
    }
}

__global__ void compute_flux_x_kernel(const float* __restrict__ s, float* __restrict__ F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ng - 1 && i < nx + ng && j >= ng && j < ny + ng) {
        float prim[4][6];
        for (int k = -2; k <= 3; ++k) {
            float rho = s[idx(0, i+k, j)];
            float inv_rho = 1.0f / rho;
            float u   = s[idx(1, i+k, j)] * inv_rho;
            float v   = s[idx(2, i+k, j)] * inv_rho;
            float E   = s[idx(3, i+k, j)];
            float P   = (gamma_air - 1.0f) * (E - 0.5f * rho * (u*u + v*v));
            prim[0][k+2] = rho; prim[1][k+2] = u; prim[2][k+2] = v; prim[3][k+2] = P;
        }
        
        float qL[4], qR[4];
        for (int d = 0; d < 4; ++d) {
            qL[d] = weno5(prim[d][0], prim[d][1], prim[d][2], prim[d][3], prim[d][4]);
            qR[d] = weno5(prim[d][5], prim[d][4], prim[d][3], prim[d][2], prim[d][1]);
        }
        
        float rhoL = qL[0], uL = qL[1], vL = qL[2], PL = qL[3];
        float inv_rhoL = 1.0f / rhoL;
        float EL = PL/(gamma_air - 1.0f) + 0.5f * rhoL * (uL*uL + vL*vL);
        float cL = sqrtf(gamma_air * PL * inv_rhoL);
        float fL[4] = {rhoL*uL, rhoL*uL*uL + PL, rhoL*uL*vL, (EL + PL)*uL};
        
        float rhoR = qR[0], uR = qR[1], vR = qR[2], PR = qR[3];
        float inv_rhoR = 1.0f / rhoR;
        float ER = PR/(gamma_air - 1.0f) + 0.5f * rhoR * (uR*uR + vR*vR);
        float cR = sqrtf(gamma_air * PR * inv_rhoR);
        float fR[4] = {rhoR*uR, rhoR*uR*uR + PR, rhoR*uR*vR, (ER + PR)*uR};
        
        float SL = fminf(uL - cL, uR - cR);
        float SR = fmaxf(uL + cL, uR + cR);
        
        if (SL >= 0.0f) {
            for (int d=0; d<4; ++d) F[idx(d, i, j)] = fL[d];
        } else if (SR <= 0.0f) {
            for (int d=0; d<4; ++d) F[idx(d, i, j)] = fR[d];
        } else {
            float SM = (PR - PL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) / (rhoL*(SL - uL) - rhoR*(SR - uR));
            if (SM >= 0.0f) {
                float U_star_L[4] = {
                    rhoL*(SL - uL)/(SL - SM),
                    rhoL*(SL - uL)/(SL - SM)*SM,
                    rhoL*(SL - uL)/(SL - SM)*vL,
                    rhoL*(SL - uL)/(SL - SM)*(EL*inv_rhoL + (SM - uL)*(SM + PL/(rhoL*(SL - uL))))
                };
                float U_L[4] = {rhoL, rhoL*uL, rhoL*vL, EL};
                for (int d=0; d<4; ++d) F[idx(d, i, j)] = fL[d] + SL * (U_star_L[d] - U_L[d]);
            } else {
                float U_star_R[4] = {
                    rhoR*(SR - uR)/(SR - SM),
                    rhoR*(SR - uR)/(SR - SM)*SM,
                    rhoR*(SR - uR)/(SR - SM)*vR,
                    rhoR*(SR - uR)/(SR - SM)*(ER*inv_rhoR + (SM - uR)*(SM + PR/(rhoR*(SR - uR))))
                };
                float U_R[4] = {rhoR, rhoR*uR, rhoR*vR, ER};
                for (int d=0; d<4; ++d) F[idx(d, i, j)] = fR[d] + SR * (U_star_R[d] - U_R[d]);
            }
        }
    }
}

__global__ void compute_flux_y_kernel(const float* __restrict__ s, float* __restrict__ G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ng && i < nx + ng && j >= ng - 1 && j < ny + ng) {
        float prim[4][6];
        for (int k = -2; k <= 3; ++k) {
            float rho = s[idx(0, i, j+k)];
            float inv_rho = 1.0f / rho;
            float u   = s[idx(1, i, j+k)] * inv_rho;
            float v   = s[idx(2, i, j+k)] * inv_rho;
            float E   = s[idx(3, i, j+k)];
            float P   = (gamma_air - 1.0f) * (E - 0.5f * rho * (u*u + v*v));
            prim[0][k+2] = rho; prim[1][k+2] = u; prim[2][k+2] = v; prim[3][k+2] = P;
        }
        
        float qL[4], qR[4];
        for (int d = 0; d < 4; ++d) {
            qL[d] = weno5(prim[d][0], prim[d][1], prim[d][2], prim[d][3], prim[d][4]);
            qR[d] = weno5(prim[d][5], prim[d][4], prim[d][3], prim[d][2], prim[d][1]);
        }
        
        float rhoL = qL[0], uL = qL[1], vL = qL[2], PL = qL[3];
        float inv_rhoL = 1.0f / rhoL;
        float EL = PL/(gamma_air - 1.0f) + 0.5f * rhoL * (uL*uL + vL*vL);
        float cL = sqrtf(gamma_air * PL * inv_rhoL);
        float fL[4] = {rhoL*vL, rhoL*uL*vL, rhoL*vL*vL + PL, (EL + PL)*vL};
        
        float rhoR = qR[0], uR = qR[1], vR = qR[2], PR = qR[3];
        float inv_rhoR = 1.0f / rhoR;
        float ER = PR/(gamma_air - 1.0f) + 0.5f * rhoR * (uR*uR + vR*vR);
        float cR = sqrtf(gamma_air * PR * inv_rhoR);
        float fR[4] = {rhoR*vR, rhoR*uR*vR, rhoR*vR*vR + PR, (ER + PR)*vR};
        
        float SL = fminf(vL - cL, vR - cR);
        float SR = fmaxf(vL + cL, vR + cR);
        
        if (SL >= 0.0f) {
            for (int d=0; d<4; ++d) G[idx(d, i, j)] = fL[d];
        } else if (SR <= 0.0f) {
            for (int d=0; d<4; ++d) G[idx(d, i, j)] = fR[d];
        } else {
            float SM = (PR - PL + rhoL*vL*(SL - vL) - rhoR*vR*(SR - vR)) / (rhoL*(SL - vL) - rhoR*(SR - vR));
            if (SM >= 0.0f) {
                float U_star_L[4] = {
                    rhoL*(SL - vL)/(SL - SM),
                    rhoL*(SL - vL)/(SL - SM)*uL,
                    rhoL*(SL - vL)/(SL - SM)*SM,
                    rhoL*(SL - vL)/(SL - SM)*(EL*inv_rhoL + (SM - vL)*(SM + PL/(rhoL*(SL - vL))))
                };
                float U_L[4] = {rhoL, rhoL*uL, rhoL*vL, EL};
                for (int d=0; d<4; ++d) G[idx(d, i, j)] = fL[d] + SL * (U_star_L[d] - U_L[d]);
            } else {
                float U_star_R[4] = {
                    rhoR*(SR - vR)/(SR - SM),
                    rhoR*(SR - vR)/(SR - SM)*uR,
                    rhoR*(SR - vR)/(SR - SM)*SM,
                    rhoR*(SR - vR)/(SR - SM)*(ER*inv_rhoR + (SM - vR)*(SM + PR/(rhoR*(SR - vR))))
                };
                float U_R[4] = {rhoR, rhoR*uR, rhoR*vR, ER};
                for (int d=0; d<4; ++d) G[idx(d, i, j)] = fR[d] + SR * (U_star_R[d] - U_R[d]);
            }
        }
    }
}

__global__ void compute_final_rhs_kernel(float* __restrict__ r, const float* __restrict__ F, const float* __restrict__ G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ng && i < nx + ng && j >= ng && j < ny + ng) {
        for (int c = 0; c < 4; ++c) {
            r[idx(c, i, j)] = - (F[idx(c, i, j)] - F[idx(c, i-1, j)]) * inv_dx
                              - (G[idx(c, i, j)] - G[idx(c, i, j-1)]) * inv_dy;
        }
    }
}

__global__ void rk3_step_kernel(int stage, float* q_out, const float* __restrict__ q_in, 
                                const float* q_orig, const float* __restrict__ rhs, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx_tot && j < ny_tot) {
        for (int c = 0; c < 4; ++c) {
            int index = idx(c, i, j);
            if (stage == 1) {
                q_out[index] = q_in[index] + dt * rhs[index];
            } else if (stage == 2) {
                q_out[index] = 0.75f * q_orig[index] + 0.25f * q_in[index] + 0.25f * dt * rhs[index];
            } else if (stage == 3) {
                q_out[index] = (1.0f/3.0f) * q_orig[index] + (2.0f/3.0f) * q_in[index] + (2.0f/3.0f) * dt * rhs[index];
            }
        }
    }
}

// =========================================================================
//                             HOST FUNCTIONS
// =========================================================================

void write_vtk_binary(int frame, const std::vector<float>& h_q) {
    const float* s = h_q.data();
    char filename[32];
    std::snprintf(filename, sizeof(filename), "khi_%04d.vtk", frame);
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    
    out << "# vtk DataFile Version 3.0\n";
    out << "KHI Simulation Data\n";
    out << "BINARY\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << nx << " " << ny << " 1\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING 1 1 1\n";
    out << "POINT_DATA " << nx * ny << "\n";
    
    out << "SCALARS Density float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float val = swap_float(s[idx(0, i, j)]);
            out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    
    out << "SCALARS Z_Vorticity float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float v_right = s[idx(2, i+1, j)] / s[idx(0, i+1, j)];
            float v_left  = s[idx(2, i-1, j)] / s[idx(0, i-1, j)];
            float u_top   = s[idx(1, i, j+1)] / s[idx(0, i, j+1)];
            float u_bot   = s[idx(1, i, j-1)] / s[idx(0, i, j-1)];
            float vort = (v_right - v_left)/(2.0f*dx) - (u_top - u_bot)/(2.0f*dy);
            float val = swap_float(vort);
            out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }

    out << "VECTORS Velocity float\n";
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float u = swap_float(s[idx(1, i, j)] / s[idx(0, i, j)]);
            float v = swap_float(s[idx(2, i, j)] / s[idx(0, i, j)]);
            float w = swap_float(0.0f);
            out.write(reinterpret_cast<const char*>(&u), sizeof(float));
            out.write(reinterpret_cast<const char*>(&v), sizeof(float));
            out.write(reinterpret_cast<const char*>(&w), sizeof(float));
        }
    }
    out.close();
}

int main() {
    std::cout << "Initializing 512x512 CUDA Benchmark (Single Precision)...\n";
    TimingModule::init_timer();
    
    int size = 4 * nx_tot * ny_tot;
    size_t bytes = size * sizeof(float);
    
    // Host Arrays
    std::vector<float> h_q(size, 0.0f);
    
    // Init condition on CPU
    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float y = (j - ng + 0.5f) * dy;
            float x = (i - ng + 0.5f) * dx;
            float rho, u;
            if (y > 0.25f && y < 0.75f) {
                u = 0.5f; rho = 2.0f;
            } else {
                u = -0.5f; rho = 1.0f;
            }
            float rand_val = dist(gen);
            float y_025 = y - 0.25f;
            float y_075 = y - 0.75f;
            float v = 0.05f * sinf(4.0f * (float)M_PI * x) * (expf(-(y_025 * y_025) / 0.005f) + expf(-(y_075 * y_075) / 0.005f)) + 0.005f * (rand_val - 0.5f);
            float P = 2.5f;
            
            h_q[idx(0, i, j)] = rho;
            h_q[idx(1, i, j)] = rho * u;
            h_q[idx(2, i, j)] = rho * v;
            h_q[idx(3, i, j)] = P/(gamma_air - 1.0f) + 0.5f * rho * (u*u + v*v);
        }
    }

    // Allocate Device Arrays
    float *d_q, *d_q1, *d_q2, *d_rhs, *d_F, *d_G, *d_max_speed;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_q1, bytes));
    CUDA_CHECK(cudaMalloc(&d_q2, bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs, bytes));
    CUDA_CHECK(cudaMalloc(&d_F, bytes));
    CUDA_CHECK(cudaMalloc(&d_G, bytes));
    CUDA_CHECK(cudaMalloc(&d_max_speed, sizeof(float)));

    // Copy Initial State to Device
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), bytes, cudaMemcpyHostToDevice));

    // Define Grid Dimensions
    dim3 threads(16, 16);   // 8 warps called
    dim3 blocks((nx_tot + threads.x - 1) / threads.x, (ny_tot + threads.y - 1) / threads.y);
    // we don't just call 512 blocks for 2 threads to strictly follow the Memory Coalescing rules as threads are called by "warp" of 32 threads, so we unsure each warp is fully utilized
    
    // Dedicated Grids for the boundary condition mapping
    dim3 bc_threads_x(ng, 16); 
    dim3 bc_blocks_x(1, (ny_tot + bc_threads_x.y - 1) / bc_threads_x.y);    // rounds down division, so we add bc_threads_x.y - 1 to ensure we cover all rows
    dim3 bc_threads_y(16, ng);
    dim3 bc_blocks_y((nx_tot + bc_threads_y.x - 1) / bc_threads_y.x, 1);    // rounds down division, so we add bc_threads_x.y - 1 to ensure we cover all rows

    // Calculate Shared Memory needed for the reduction kernel
    int shared_mem_bytes = threads.x * threads.y * sizeof(float);

    float t = 0.0f, t_next_output = 0.0f;
    int frame = 0;
    
    // Initial BCs at t=0
    apply_bcs_x_kernel<<<bc_blocks_x, bc_threads_x>>>(d_q);
    apply_bcs_y_kernel<<<bc_blocks_y, bc_threads_y>>>(d_q);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    while (t < t_max) {
        if (t >= t_next_output - 1.0e-8f) {
            TimingModule::start_io_timer();
            std::cout << "Time: " << t << " / " << t_max << " | Frame: " << frame << "\n";
            
            CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, bytes, cudaMemcpyDeviceToHost));
            write_vtk_binary(frame, h_q);
            
            TimingModule::stop_io_timer();
            frame++;
            t_next_output += output_interval;
        }
        
        TimingModule::start_math_timer();
        
        // 1. Direct memset resets the global max speed value without CPU overhead
        CUDA_CHECK(cudaMemset(d_max_speed, 0, sizeof(float)));
        
        // 2. Launch reduction kernel to compute new max speed
        compute_dt_kernel<<<blocks, threads, shared_mem_bytes>>>(d_q, d_max_speed);
        
        // 3. Bring the single float back to CPU
        float h_max_speed = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_max_speed, d_max_speed, sizeof(float), cudaMemcpyDeviceToHost));
        h_max_speed = fmaxf(h_max_speed, 1.0e-8f);
        
        float dt = CFL_target * fminf(dx, dy) / h_max_speed;
        if (t + dt > t_next_output) dt = t_next_output - t;

        // Stage 1
        apply_bcs_x_kernel<<<bc_blocks_x, bc_threads_x>>>(d_q);
        apply_bcs_y_kernel<<<bc_blocks_y, bc_threads_y>>>(d_q);
        compute_flux_x_kernel<<<blocks, threads>>>(d_q, d_F);
        compute_flux_y_kernel<<<blocks, threads>>>(d_q, d_G);
        compute_final_rhs_kernel<<<blocks, threads>>>(d_rhs, d_F, d_G);
        rk3_step_kernel<<<blocks, threads>>>(1, d_q1, d_q, d_q, d_rhs, dt);

        // Stage 2
        apply_bcs_x_kernel<<<bc_blocks_x, bc_threads_x>>>(d_q1);
        apply_bcs_y_kernel<<<bc_blocks_y, bc_threads_y>>>(d_q1);
        compute_flux_x_kernel<<<blocks, threads>>>(d_q1, d_F);
        compute_flux_y_kernel<<<blocks, threads>>>(d_q1, d_G);
        compute_final_rhs_kernel<<<blocks, threads>>>(d_rhs, d_F, d_G);
        rk3_step_kernel<<<blocks, threads>>>(2, d_q2, d_q1, d_q, d_rhs, dt);

        // Stage 3
        apply_bcs_x_kernel<<<bc_blocks_x, bc_threads_x>>>(d_q2);
        apply_bcs_y_kernel<<<bc_blocks_y, bc_threads_y>>>(d_q2);
        compute_flux_x_kernel<<<blocks, threads>>>(d_q2, d_F);
        compute_flux_y_kernel<<<blocks, threads>>>(d_q2, d_G);
        compute_final_rhs_kernel<<<blocks, threads>>>(d_rhs, d_F, d_G);
        rk3_step_kernel<<<blocks, threads>>>(3, d_q, d_q2, d_q, d_rhs, dt);

        // Force CPU to wait for GPU to finish this timestep before pausing timer
        CUDA_CHECK(cudaDeviceSynchronize());
        
        TimingModule::stop_math_timer();
        t += dt;
    }
    
    cudaFree(d_q); cudaFree(d_q1); cudaFree(d_q2);
    cudaFree(d_rhs); cudaFree(d_F); cudaFree(d_G); cudaFree(d_max_speed);
    
    TimingModule::print_timing_results(nx, ny);
    return 0;
}