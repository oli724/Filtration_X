import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import constants
import sys
import os
from MCA_parser import MCA

# Configuration
TARGET_ENERGY = 10  # keV
ENERGY_TOLERANCE = 0.5  # ±0.5 keV window
DATA_DIR = "semaine_1"
REF_FILE = os.path.join(DATA_DIR, "Courant_20kV_25uA.mca")

# Material properties (Z, density g/cm³, files pattern)
materials = {
    'Cu': {'Z': 29, 'rho': 8.96, 'pattern': 'Cu'},
    'Ag': {'Z': 47, 'rho': 10.49, 'pattern': 'Ag'},
    'Mo': {'Z': 42, 'rho': 10.2, 'pattern': 'Mo'},
    'W': {'Z': 74, 'rho': 19.25, 'pattern': 'W'},
    'Al': {'Z': 13, 'rho': 2.7, 'pattern': 'Al'}
}

# Conversion factor (10⁻²⁴ cm²)
CONV_FACTOR = 1e-24

def get_energy_index(energies, target_energy, tolerance):
    """Find index closest to target energy"""
    return np.where(np.abs(energies - target_energy) <= tolerance)[0]

# Load reference spectrum
mca_N0 = MCA(REF_FILE)
energies = np.load("energie_semaine_1.npy")  # Your energy calibration
ref_counts = np.array(mca_N0.DATA)
ref_livetime = mca_N0.get_live_time()
ref_rates = ref_counts / ref_livetime

# Find energy bin for 12 keV
energy_idx = get_energy_index(energies, TARGET_ENERGY, ENERGY_TOLERANCE)
if len(energy_idx) == 0:
    raise ValueError(f"No energy bins found near {TARGET_ENERGY} keV")

# Process each material
results = []
for mat, props in materials.items():
    mat_files = [f for f in os.listdir(DATA_DIR) 
                if f.startswith(props['pattern']) and f.endswith('.mca')]
    
    if not mat_files:
        print(f"No files found for {mat}")
        continue

    tau_values = []
    for file in mat_files:
        try:
            # Load sample spectrum
            mca = MCA(os.path.join(DATA_DIR, file))
            sample_counts = np.array(mca.DATA)
            livetime = mca.get_live_time()
            sample_rates = sample_counts / livetime

            # Calculate τ for each matching energy bin
            for idx in energy_idx:
                if ref_rates[idx] > 0 and sample_rates[idx] > 0:
                    N_N0 = sample_rates[idx] / ref_rates[idx]
                    t_mils = float(file.split('_')[1].replace('mils', ''))
                    t_cm = t_mils * 0.00254
                    
                    # τ in cm⁻¹ then convert to cm²/atom
                    tau_cm = (-np.log(N_N0)/(t_cm * props['rho'])) 
                    tau_cm2 = tau_cm * props['Z']/constants.N_A  # Convert to cm²/atom
                    tau_values.append(tau_cm2)

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if tau_values:
        tau_mean = np.mean(tau_values)
        results.append({
            'material': mat,
            'Z': props['Z'],
            'lnZ': np.log(props['Z']),
            'ln_tau': np.log(tau_mean / CONV_FACTOR),
            'tau_values': tau_values  # For error analysis
        })
        print(f"{mat}: tau = {tau_mean:.3e} cm2/atom at {TARGET_ENERGY} keV")

# Perform linear regression on ln(τ) vs ln(Z)
if len(results) >= 2:
    Z = np.array([r['Z'] for r in results])
    lnZ = np.array([r['lnZ'] for r in results])
    ln_tau = np.array([r['ln_tau'] for r in results])
    
    # Calculate errors (standard error of the mean)
    ln_tau_err = [np.std(np.log(np.array(r['tau_values'])/CONV_FACTOR))/np.sqrt(len(r['tau_values'])) 
                 for r in results]
    
    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(lnZ, ln_tau)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Data points with error bars
    plt.errorbar(lnZ, ln_tau, yerr=ln_tau_err, fmt='o', markersize=8,
                capsize=5, label='Experimental Data')
    
    # Fit line
    fit_x = np.linspace(min(lnZ), max(lnZ), 100)
    fit_y = slope * fit_x + intercept
    plt.plot(fit_x, fit_y, 'r--', 
             label=f'Fit: ln(tau) = {slope:.2f}ln(Z) + {intercept:.2f}\n(R² = {r_value**2:.3f})')
    
    # Annotate points
    for r in results:
        plt.annotate(r['material'], (r['lnZ'], r['ln_tau']), 
                    xytext=(5,5), textcoords='offset points')
    
    plt.xlabel('ln(Z)')
    plt.ylabel(f'ln(τ/10-24 cm2) at {TARGET_ENERGY} keV')
    plt.title('Photoelectric Effect: ln(tau) vs ln(Z)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Print theoretical expectation (τ ∝ Z^n where n≈4-5)
    print(f"\nFit results at {TARGET_ENERGY} keV:")
    print(f"Slope (n): {slope:.2f} \pm {std_err:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print(f"Expected slope for photoelectric effect: ~4-5")
    
    plt.show()
else:
    print("Insufficient data for analysis")