import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
RESULTS_DIR = "results"
CSV_FILE = os.path.join(RESULTS_DIR, "benchmarks.csv")
NX, NY = 512, 512  # Adjust if you run different benchmarks

def read_vtk_density(filepath, nx, ny):
    """Extract Density data from binary VTK format."""
    try:
        with open(filepath, 'rb') as f:
            # Skip the ASCII header until we hit the binary Density data
            for line in f:
                if b"LOOKUP_TABLE default" in line:
                    break
            # Read exactly nx * ny floats (big-endian format '>f4')
            density_data = np.fromfile(f, dtype='>f4', count=nx*ny)
            return density_data
    except FileNotFoundError:
        print(f"  [!] Missing file: {filepath}")
        return None

def plot_benchmarks():
    """Reads benchmarks.csv and generates high-quality performance analytics."""
    print("\n--- 1. Generating Performance Analytics ---")
    if not os.path.exists(CSV_FILE):
        print(f"  [!] {CSV_FILE} not found. Run your solvers first.")
        return

    # Load data and clean column names
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    
    # Focus on the largest grid for comparison
    max_grid = df['GridSize'].max()
    df_filtered = df[df['GridSize'] == max_grid].copy()
    
    # Identify Baseline (Serial) for Speedup calculation
    # We find the row where Architecture is 'Serial'
    baseline_row = df_filtered[df_filtered['Architecture'].str.contains('Serial_Fortran', case=False)]
    if baseline_row.empty:
        baseline_time = df_filtered.iloc[0]['MathTime_sec']
        print("  [i] 'Serial_Fortran' not found, using first row as baseline.")
    else:
        baseline_time = baseline_row.iloc[0]['MathTime_sec']

    df_filtered['Speedup'] = baseline_time / df_filtered['MathTime_sec']

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-muted') # Clean, modern look
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['#457b9d', '#1d3557', '#e63946', '#2a9d8f', '#f4a261']

    # Plot 1: MCUPS (Throughput)
    bars1 = ax1.bar(df_filtered['Architecture'], df_filtered['MCUPS'], color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title(f'Solver Throughput ({max_grid}x{max_grid})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MCUPS (Higher is Better)', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Plot 2: Speedup and Math Time
    bars2 = ax2.bar(df_filtered['Architecture'], df_filtered['MathTime_sec'], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_title('Execution Latency & Speedup', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Seconds (Lower is Better)', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Add Text Annotations ---
    for i, bar in enumerate(bars1):
        # Label for MCUPS
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                 f'{df_filtered.iloc[i]["MCUPS"]:.1f}', ha='center', fontweight='bold')
        
        # Label for Speedup (on the time plot)
        speedup = df_filtered.iloc[i]['Speedup']
        label = "Baseline" if speedup == 1.0 else f"{speedup:.1f}x"
        ax2.text(bars2[i].get_x() + bars2[i].get_width()/2, bars2[i].get_height() * 1.02, 
                 label, ha='center', color='#d62828', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('KHI Solver Performance Architecture Comparison', fontsize=16)

    save_path = os.path.join(RESULTS_DIR, "performance_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"  [+] Chart saved to: {save_path}")

def validate_physics(frame_num=40):
    """Compares the final VTK frames against the Serial baseline."""
    print("\n--- 2. Validating Physics (RMSE) ---")
    
    # Assuming FORTRAN Serial is our ground truth
    baseline_path = os.path.join(RESULTS_DIR, "FORTRAN_VTK", f"khi_{frame_num:04d}.vtk")
    baseline_data = read_vtk_density(baseline_path, NX, NY)
    
    if baseline_data is None:
        print("  [!] Baseline VTK not found. Cannot perform validation.")
        return

    # Compare other architectures to the baseline
    test_folders = ["OMP_VTK", "ACC_VTK", "C++_VTK", "CUDA_VTK"]
    
    for folder in test_folders:
        test_path = os.path.join(RESULTS_DIR, folder, f"khi_{frame_num:04d}.vtk")
        if not os.path.exists(test_path):
            continue
            
        test_data = read_vtk_density(test_path, NX, NY)
        
        # Calculate Root Mean Square Error
        # $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2}$
        mse = np.mean((baseline_data - test_data)**2)
        rmse = np.sqrt(mse)
        
        # A tiny floating point drift (e.g., 1e-6) is totally normal between CPU and GPU math
        status = "PASS" if rmse < 1e-4 else "FAIL"
        print(f"  [{status}] {folder:12} vs Baseline -> RMSE: {rmse:.2e}")

if __name__ == "__main__":
    plot_benchmarks()
    validate_physics(frame_num=1) # first frames to prevent butterfly effect