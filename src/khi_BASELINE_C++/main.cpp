#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <timing.hpp>

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

// --- Flat 1D Array Indexing (Structure of Arrays - SoA layout) --- SoA layout better once ported in CUDA for coalesced memory access
inline int idx(int c, int i, int j) {
    return (i + j * nx_tot) + c * (nx_tot * ny_tot);
}

// --- Endian Swapping for Binary VTK ---
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

// --- WENO5 Reconstruction (Optimized with Fast Multiplication) ---
inline float weno5(float v1, float v2, float v3, float v4, float v5) {
    const float eps = 1.0e-6;
    
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

// --- Apply Periodic Boundaries ---
void apply_periodic_bcs(std::vector<float>& state) {
    float* __restrict__ s = state.data();
    
    // X-boundaries: k is acting as the fast 'i' index
    for (int c = 0; c < 4; ++c) {
        for (int j = 0; j < ny_tot; ++j) {
            for (int k = 0; k < ng; ++k) {
                s[idx(c, k, j)] = s[idx(c, nx + k, j)];
                s[idx(c, nx + ng + k, j)] = s[idx(c, ng + k, j)];
            }
        }
    }
    
    // Y-boundaries: i is the fast index, k is acting as the slow 'j' index
    for (int c = 0; c < 4; ++c) {
        for (int k = 0; k < ng; ++k) {
            for (int i = 0; i < nx_tot; ++i) {
                s[idx(c, i, k)] = s[idx(c, i, ny + k)];
                s[idx(c, i, ny + ng + k)] = s[idx(c, i, ng + k)];
            }
        }
    }
}
// --- Compute Adaptive Time-Step ---
float compute_dt(const std::vector<float>& state) {
    const float* __restrict__ s = state.data();
    float max_speed = 1.0e-8;
    
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float rho = s[idx(0, i, j)];
            float u   = s[idx(1, i, j)] / rho;
            float v   = s[idx(2, i, j)] / rho;
            float P   = (gamma_air - 1.0f) * (s[idx(3, i, j)] - 0.5f * rho * (u*u + v*v));
            float c   = std::sqrt(gamma_air * P / rho);
            max_speed  = std::max({max_speed, std::abs(u) + c, std::abs(v) + c});
        }
    }
    return CFL_target * std::min(dx, dy) / max_speed;
}

// --- Compute RHS (WENO5 + HLLC) ---
void compute_rhs(const std::vector<float>& state, std::vector<float>& rhs, 
                 std::vector<float>& F_ext, std::vector<float>& G_ext) {
    
    const float* __restrict__ s = state.data();
    float* __restrict__ r = rhs.data();
    float* __restrict__ F = F_ext.data();
    float* __restrict__ G = G_ext.data();
    
    // --- X-Direction Fluxes ---
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng - 1; i < nx + ng; ++i) { 
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
            float cL = std::sqrt(gamma_air * PL * inv_rhoL);
            float fL[4] = {rhoL*uL, rhoL*uL*uL + PL, rhoL*uL*vL, (EL + PL)*uL};
            
            float rhoR = qR[0], uR = qR[1], vR = qR[2], PR = qR[3];
            float inv_rhoR = 1.0f / rhoR;
            float ER = PR/(gamma_air - 1.0f) + 0.5f * rhoR * (uR*uR + vR*vR);
            float cR = std::sqrt(gamma_air * PR * inv_rhoR);
            float fR[4] = {rhoR*uR, rhoR*uR*uR + PR, rhoR*uR*vR, (ER + PR)*uR};
            
            float SL = std::min(uL - cL, uR - cR);
            float SR = std::max(uL + cL, uR + cR);
            
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
    
    // --- Y-Direction Fluxes ---
    for (int j = ng - 1; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) { 
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
            float cL = std::sqrt(gamma_air * PL * inv_rhoL);
            float fL[4] = {rhoL*vL, rhoL*uL*vL, rhoL*vL*vL + PL, (EL + PL)*vL};
            
            float rhoR = qR[0], uR = qR[1], vR = qR[2], PR = qR[3];
            float inv_rhoR = 1.0f / rhoR;
            float ER = PR/(gamma_air - 1.0f) + 0.5f * rhoR * (uR*uR + vR*vR);
            float cR = std::sqrt(gamma_air * PR * inv_rhoR);
            float fR[4] = {rhoR*vR, rhoR*uR*vR, rhoR*vR*vR + PR, (ER + PR)*vR};
            
            float SL = std::min(vL - cL, vR - cR);
            float SR = std::max(vL + cL, vR + cR);
            
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
    
// --- Compute Final RHS ---
    for (int c = 0; c < 4; ++c) {
        for (int j = ng; j < ny + ng; ++j) {
            for (int i = ng; i < nx + ng; ++i) {
                r[idx(c, i, j)] = - (F[idx(c, i, j)] - F[idx(c, i-1, j)]) * inv_dx
                                  - (G[idx(c, i, j)] - G[idx(c, i, j-1)]) * inv_dy;
            }
        }
    }
}

// --- Binary VTK Writer ---
void write_vtk_binary(int frame, const std::vector<float>& state) {
    const float* __restrict__ s = state.data();
    
    char filename[32];
    std::snprintf(filename, sizeof(filename), "khi_%04d.vtk", frame);
    
    std::ofstream out(filename, std::ios::out | std::ios::binary);
   
    // Header
    out << "# vtk DataFile Version 3.0\n";
    out << "KHI Simulation Data\n";
    out << "BINARY\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << nx << " " << ny << " 1\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING 1 1 1\n";
    out << "POINT_DATA " << nx * ny << "\n";
    

    // 1. Density
    out << "SCALARS Density float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float val = swap_float(static_cast<float>(s[idx(0, i, j)]));
            out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    

    // 2. Z-Vorticity
    out << "SCALARS Z_Vorticity float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float v_right = s[idx(2, i+1, j)] / s[idx(0, i+1, j)];
            float v_left  = s[idx(2, i-1, j)] / s[idx(0, i-1, j)];
            float u_top   = s[idx(1, i, j+1)] / s[idx(0, i, j+1)];
            float u_bot   = s[idx(1, i, j-1)] / s[idx(0, i, j-1)];
            
            float vort = (v_right - v_left)/(2.0f*dx) - (u_top - u_bot)/(2.0f*dy);
            float val = swap_float(static_cast<float>(vort));
            out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }

    // 3. Velocity Vectors 
    out << "VECTORS Velocity float\n";
    for (int j = ng; j < ny + ng; ++j) {
        for (int i = ng; i < nx + ng; ++i) {
            float u = swap_float(static_cast<float>(s[idx(1, i, j)] / s[idx(0, i, j)]));
            float v = swap_float(static_cast<float>(s[idx(2, i, j)] / s[idx(0, i, j)]));
            float w = swap_float(0.0f);
            
            out.write(reinterpret_cast<const char*>(&u), sizeof(float));
            out.write(reinterpret_cast<const char*>(&v), sizeof(float));
            out.write(reinterpret_cast<const char*>(&w), sizeof(float));
        }
    }
    out.close();
}

int main() {
    std::cout << "Initializing 512x512 Pure Serial C++ Benchmark...\n";
    TimingModule::init_timer();
    
    int size = 4 * nx_tot * ny_tot;
    std::vector<float> q(size, 0.0f), q1(size, 0.0f), q2(size, 0.0f);
    std::vector<float> rhs(size, 0.0f), F_ext(size, 0.0f), G_ext(size, 0.0f);
    
    float* __restrict__ q_ptr = q.data();
    float* __restrict__ q1_ptr = q1.data();
    float* __restrict__ q2_ptr = q2.data();
    float* __restrict__ rhs_ptr = rhs.data();
    
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
            
            float y_025 = y - 0.25f;
            float y_075 = y - 0.75f;
            float v = 0.05f * std::sin(4.0f * (float)M_PI * x) * (std::exp(-(y_025 * y_025) / 0.005f) + std::exp(-(y_075 * y_075) / 0.005f));            
            float P = 2.5f;
            
            q_ptr[idx(0, i, j)] = rho;
            q_ptr[idx(1, i, j)] = rho * u;
            q_ptr[idx(2, i, j)] = rho * v;
            q_ptr[idx(3, i, j)] = P/(gamma_air - 1.0f) + 0.5f * rho * (u*u + v*v);
        }
    }
    
    float t = 0.0f, t_next_output = 0.0f;
    int frame = 0;
    
    apply_periodic_bcs(q); 
    
    while (t < t_max) {
        
        // --- Output Stage ---
        if (t >= t_next_output - 1.0e-8f) {
            TimingModule::start_io_timer();
            std::cout << "Time: " << t << " / " << t_max << " | Frame: " << frame << "\n";
            write_vtk_binary(frame, q);
            TimingModule::stop_io_timer();
            
            frame++;
            t_next_output += output_interval;
        }
        
        // --- Math Stage ---
        TimingModule::start_math_timer();
        
        float dt = compute_dt(q);
        if (t + dt > t_next_output) dt = t_next_output - t;
        
        // SSP-RK3 Stage 1
        apply_periodic_bcs(q);
        compute_rhs(q, rhs, F_ext, G_ext);
        for (int c = 0; c < 4; ++c) {
            for (int j = 0; j < ny_tot; ++j) {
                for (int i = 0; i < nx_tot; ++i) {
                    q1_ptr[idx(c, i, j)] = q_ptr[idx(c, i, j)] + dt * rhs_ptr[idx(c, i, j)];
                }
            }
        }
        
        // SSP-RK3 Stage 2
        apply_periodic_bcs(q1);
        compute_rhs(q1, rhs, F_ext, G_ext);
        for (int c = 0; c < 4; ++c) {
            for (int j = 0; j < ny_tot; ++j) {
                for (int i = 0; i < nx_tot; ++i) {
                    q2_ptr[idx(c, i, j)] = 0.75f * q_ptr[idx(c, i, j)] + 0.25f * q1_ptr[idx(c, i, j)] + 0.25f * dt * rhs_ptr[idx(c, i, j)];
                }
            }
        }
        
        // SSP-RK3 Stage 3
        apply_periodic_bcs(q2);
        compute_rhs(q2, rhs, F_ext, G_ext);
        for (int c = 0; c < 4; ++c) {
            for (int j = 0; j < ny_tot; ++j) {
                for (int i = 0; i < nx_tot; ++i) {
                    q_ptr[idx(c, i, j)] = (1.0f/3.0f) * q_ptr[idx(c, i, j)] + (2.0f/3.0f) * q2_ptr[idx(c, i, j)] + (2.0f/3.0f) * dt * rhs_ptr[idx(c, i, j)];
                }
            }
        }
        
        TimingModule::stop_math_timer();
        t += dt;
    }
    
    TimingModule::print_timing_results(nx, ny, "Serial_C++");
    return 0;
}