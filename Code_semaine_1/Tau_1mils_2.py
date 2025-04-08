import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
import sys
import os
from pathlib import Path

# Ajouter le dossier parent au chemin Python
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from MCA_parser import MCA


sys.stdout.reconfigure(encoding='utf-8')
energies= np.load("energie_semaine_1.npy")
# Fonction pour calculer tau
# Main analysis code
def calculate_tau(N, N0, t_cm, rho, A):
    N_A = constants.N_A
    return (-np.log(N/N0)/(rho*t_cm)) * (A/N_A) - 0.20*(A/N_A)

def safe_print(*args, **kwargs):
    """Wrapper for print that handles encoding issues"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding
        args = [str(arg).encode(encoding, errors='replace').decode(encoding) 
            if isinstance(arg, str) else arg for arg in args]
        print(*args, **kwargs)

# Material properties
materials = {
    'Cu': {'Z': 29, 'rho': 8.96, 'pattern': 'Cu'},
    'Ag': {'Z': 47, 'rho': 10.49, 'pattern': 'Ag'},
    'Mo': {'Z': 42, 'rho': 10.28, 'pattern': 'Mo'},
    'W': {'Z': 74, 'rho': 19.25, 'pattern': 'W'},
    'Al': {'Z': 13, 'rho': 2.7, 'pattern': 'Al'}
}

# Process files
data_dir = "semaine_1"
results = {}

# Load reference spectrum
try:
    ref_file = os.path.join(data_dir, "Courant_20kV_25uA.mca")
    mca_N0 = MCA(ref_file)
    N0_counts = np.array(mca_N0.DATA)
    N0_live_time = mca_N0.get_live_time()
    N0_count_rate = N0_counts / N0_live_time
except Exception as e:
    safe_print(f"Error loading reference file: {e}")
    sys.exit(1)

# Process each material
for mat, props in materials.items():
    try:
        files = [f for f in os.listdir(data_dir) 
                if f.startswith(props['pattern']) and f.endswith('.mca')]
        
        if not files:
            safe_print(f"No files found for {mat}")
            continue

        tau_values = []
        for file in files:
            try:
                # Extract thickness (in mils)
                parts = file.split('_')
                t_mils = float(parts[1].replace('mils', ''))
                
                # Skip if:
                # - Not 1 mil (for non-Al) or not 10 mil (for Al)
                if (mat.lower() != 'Al' and t_mils != 1) or \
                   (mat.lower() == 'Al' and t_mils != 10):
                    continue
                
                t_cm = t_mils * 0.00254  # Convert to cm

                # Process spectrum
                mca = MCA(os.path.join(data_dir, file))
                counts = np.array(mca.DATA)
                live_time = mca.get_live_time()
                count_rate = counts / live_time

                # Apply energy filter (e.g., 10-18 keV)
                energy_mask = (energies >= 0) & (energies <= 20)
                filtered_N0 = N0_count_rate[energy_mask]
                filtered_counts = count_rate[energy_mask]

                # Calculate ratio (skip invalid values)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where((filtered_N0 > 0) & (filtered_counts > 0),
                                    filtered_counts / filtered_N0,
                                    np.nan)
                
                valid_ratio = ratio[np.isfinite(ratio)]
                if len(valid_ratio) > 0:
                    tau = calculate_tau(valid_ratio.mean(), 1, t_cm, props['rho'], props['Z'])
                    tau_values.append(tau)
                    safe_print(f"Processed {file} (t={t_mils} mils) - τ = {tau:.3e}")
                else:
                    safe_print(f"Invalid ratio for {file} in selected energy range")

            except Exception as e:
                safe_print(f"Error processing {file}: {e}")
                continue

        if tau_values:
            results[mat] = {
                'Z': props['Z'],
                'tau_mean': np.mean(tau_values),
                'tau_std': np.std(tau_values),
                'n_samples': len(tau_values)
            }
    except Exception as e:
        safe_print(f"Error processing {mat}: {e}")
        continue

# Analysis and plotting
if results:
    # Prepare data
    Z = np.array([data['Z'] for data in results.values()])
    tau = np.array([data['tau_mean'] for data in results.values()])
    tau_err = np.array([data['tau_std'] for data in results.values()])
    
    # Convert to log scale (with τ scaled by 1e-24 for better visualization)
    log_Z = np.log(Z)
    log_tau = np.log(tau/1e-24)
    log_tau_err = tau_err/tau  # Error propagation for log scale

    # Linear fit function: ln(τ/1e-24) = a + b*ln(Z)
    def linear_fit(log_Z, a, b):
        return a + b*log_Z

    try:
        # Perform linear fit in log space
        popt, pcov = curve_fit(linear_fit, log_Z, log_tau)
                              #sigma=log_tau_err, absolute_sigma=True)
        a_fit, b_fit = popt
        b_err = np.sqrt(pcov[1,1])  # Uncertainty in slope
        
        # Generate fit line
        Z_fit = np.linspace(min(Z), max(Z), 100)
        log_Z_fit = np.log(Z_fit)
        log_tau_fit = linear_fit(log_Z_fit, a_fit, b_fit)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        
        # Plot data points
        plt.errorbar(Z, tau/1e-24, yerr=tau_err/1e-24, 
                    fmt='o', capsize=5, label='Experimental Data')
        
        # Plot fit line
        plt.plot(Z_fit, np.exp(log_tau_fit), 'r-', 
                label=f'Fit: τ ∝ Z$^{{{b_fit:.2f}±{b_err:.2f}}}$')
        
        # Format as log-log plot
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Atomic Number (Z)', fontsize=12)
        plt.ylabel('τ (10$^{-24}$ cm$^{-1}$)', fontsize=12)
        plt.title('τ vs Z (Log-Log Scale)', fontsize=14)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.legend()
        
        # Print fit results
        safe_print(f"\nFit results:")
        safe_print(f"Slope (exponent) = {b_fit:.2f} ± {b_err:.2f}")
        safe_print(f"Expected slope ≈4 for photoelectric dominance")
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        safe_print(f"Fit error: {e}")
        # Fallback plot if fit fails
        plt.figure(figsize=(8, 6))
        plt.scatter(Z, tau/1e-24)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Z')
        plt.ylabel('τ (10$^{-24}$ cm$^{-1}$)')
        plt.title('Data (Fit Failed)')
        plt.show()
        

    # Print results
    safe_print("\nFinal Results:")
    safe_print("{:<5} {:<8} {:<12} {:<12} {:<10}".format(
        'Mat', 'Z', 'tau_mean', 'tau_std', 'N_samples'))
    for mat, data in results.items():
        safe_print("{:<5} {:<8} {:<12.3e} {:<12.3e} {:<10}".format(
            mat, data['Z'], data['tau_mean'], data['tau_std'], data['n_samples']))
else:
    safe_print("No valid results obtained from any materials")