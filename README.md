# Kelvin-Helmholtz Instability Solvers

Simulates the Kelvin-Helmholtz Instability across different languages (Fortran, C++) and computing paradigms (OpenMP, OpenACC, CUDA) to compare performance and computation times.

<img width="800" height="800" alt="khi_small" src="https://github.com/user-attachments/assets/491686bf-c3b6-49b4-b8ab-abf670cac397" />

## Physics
This project simulates the Kelvin-Helmholtz Instability (KHI), a fluid dynamics phenomenon that occurs at the interface of two fluids with different densities and shear velocities. 

To capture high-order instabilities and fine-scale turbulent structures, the simulation solves the compressible **Euler Equations**. System closure is achieved using the Total Energy Density equation for a Calorically Perfect Gas.

## Numerical Schemes
To ensure a high-fidelity simulation while strictly preserving numerical stability, the solver implements the following advanced CFD schemes:
* **Spatial Reconstruction:** WENO5 (5th-Order Weighted Essentially Non-Oscillatory)
* **Riemann Solver:** HLLC (Harten-Lax-van Leer-Contact)
* **Time Integration:** SSP-RK3 (3rd-Order Strong Stability Preserving Runge-Kutta)

## Code Architecture
The solver was progressively developed and ported to evaluate different hardware acceleration strategies:
* **Fortran:** Developed as the baseline serial implementation, followed by CPU multi-threading via **OpenMP** and GPU acceleration via **OpenACC**.
* **C++:** Ported for an object-oriented serial baseline, then fundamentally restructured for highly optimized, native GPU execution using **CUDA**.

## Results & Benchmarks
Physics accuracy is validated by calculating the Root Mean Square Error (RMSE) against the Fortran serial baseline. 

<img width="100%" alt="performance_comparison" src="https://github.com/user-attachments/assets/3bc9e0fe-f451-43a1-9fd1-72825086d693" />

```text
--- 2. Validating Physics (RMSE) ---
  [PASS] OMP_VTK      vs Baseline -> RMSE: 1.19e-06
  [PASS] ACC_VTK      vs Baseline -> RMSE: 1.65e-06
  [PASS] C++_VTK      vs Baseline -> RMSE: 4.03e-04
  [PASS] CUDA_VTK     vs Baseline -> RMSE: 2.99e-06
