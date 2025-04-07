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
energies = np.load("energie_semaine_1.npy")

# Fonction pour calculer tau
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

# Material properties with ATOMIC MASSES (A) instead of Z
materials = {
    'Al': {'Z': 13, 'A': 26.982, 'rho': 2.7, 'pattern': 'Al'},
    'Cu': {'Z': 29, 'A': 63.546, 'rho': 8.96, 'pattern': 'Cu'},
    'Mo': {'Z': 42, 'A': 95.95, 'rho': 10.28, 'pattern': 'Mo'},
    'Ag': {'Z': 47, 'A': 107.868, 'rho': 10.49, 'pattern': 'Ag'},
    'W': {'Z': 74, 'A': 183.84, 'rho': 19.25, 'pattern': 'W'}
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

# Process each material with 15.9-16 keV filter
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
                # Extract thickness
                parts = file.split('_')
                t_mils = float(parts[1].replace('mils', ''))
                t_cm = t_mils * 0.00254

                # Process spectrum
                mca = MCA(os.path.join(data_dir, file))
                counts = np.array(mca.DATA)
                live_time = mca.get_live_time()
                count_rate = counts / live_time

                # Apply energy filter (15.9-16 keV)
                energy_mask = (energies >= 14.95) & (energies <= 15.05)
                filtered_N0 = N0_count_rate[energy_mask]
                filtered_counts = count_rate[energy_mask]

                # Calculate ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where((filtered_N0 > 0) & (filtered_counts > 0),
                                    filtered_counts / filtered_N0,
                                    np.nan)
                
                valid_ratio = ratio[np.isfinite(ratio)]
                if len(valid_ratio) > 0:
                    # Using ATOMIC MASS (A) instead of Z here
                    tau = calculate_tau(valid_ratio.mean(), 1, t_cm, props['rho'], props['A'])
                    tau_values.append(tau)
                else:
                    safe_print(f"Invalid ratio for {file} in 15.9-16 keV range")

            except Exception as e:
                safe_print(f"Error processing {file}: {e}")
                continue

        if tau_values:
            results[mat] = {
                'Z': props['Z'],
                'A': props['A'],
                'tau_mean': np.mean(tau_values),
                'tau_std': np.std(tau_values),
                'n_samples': len(tau_values)
            }
    except Exception as e:
        safe_print(f"Error processing {mat}: {e}")
        continue

# Analysis and plotting
if results:
    # Prepare data - we'll plot against Z but calculate using A
    Z = np.array([data['Z'] for data in results.values()])
    A = np.array([data['A'] for data in results.values()])
    tau = np.array([data['tau_mean'] for data in results.values()])
    tau_err = np.array([data['tau_std'] for data in results.values()])
    
    # Convert to log scale
    log_Z = np.log(Z)
    log_tau = np.log(tau/1e-24)
    log_tau_err = tau_err/tau  # Error propagation

    # Linear fit function
    def linear_fit(log_Z, a, b):
        return a + b*log_Z

    try:
        # Perform fit
        popt, pcov = curve_fit(linear_fit, log_Z, log_tau)
        a_fit, b_fit = popt
        b_err = np.sqrt(pcov[1,1])
        
        # Generate fit curve
        Z_fit = np.linspace(min(Z), max(Z), 100)
        log_Z_fit = np.log(Z_fit)
        log_tau_fit = linear_fit(log_Z_fit, a_fit, b_fit)
        
        # NIST reference data (must match Z order from materials dictionary)
        nist_Z = np.array([13, 29, 42, 47, 74])
        nist_tau = np.array([3.36707058e-22, 2.84905662e-21, 4.30187553e-21, 1.16675533e-20, 2.43422759e-20])/1e-24
        
        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot experimental data
        plt.errorbar(Z, tau/1e-24, yerr=tau_err/1e-24, 
                    fmt='o', capsize=5, markersize=20, label='Données expérimentales')

        # Plot NIST reference data 
        plt.scatter(nist_Z, nist_tau, color='r', s=200, marker="^", label="Données du NIST")

        # Plot fits
        plt.plot(Z_fit, np.exp(log_tau_fit), 'b--', 
                label=f'Ajustement: τ ∝ Z$^{{{b_fit:.2f}±{b_err:.2f}}}$')
        plt.plot(Z_fit, np.exp(linear_fit(log_Z_fit, -0.49278, 2.47244)), 'r:', 
                label='Ajustement NIST: τ ∝ Z$^{2.47±0.25}$')

        # Add element labels to experimental points
        for z, t, elem in zip(Z, tau/1e-24, materials.keys()):
            plt.text(z, t, f' {elem}', 
                    fontsize=25, va='center', ha='left')

        # Add element labels to NIST points
        for z, t, elem in zip(nist_Z, nist_tau, materials.keys()):
            plt.text(z, t, f' {elem}', 
                    fontsize=25, va='center', ha='left', color='r')
        
        # Calculate and print differences
        percent_diff = []
        for elem in materials:
            z = materials[elem]['Z']
            exp_idx = np.where(Z == z)[0][0]
            nist_idx = np.where(nist_Z == z)[0][0]
            diff = ((tau[exp_idx]*1e24 - nist_tau[nist_idx])/nist_tau[nist_idx])*100
            percent_diff.append(diff)
            safe_print(f"{elem}: {diff:.1f}% difference")
        print(tau/1e-24)
        print(tau_err/1e-24)
        print(nist_tau)
        # Formatting
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('log(Z)', fontsize=30)
        plt.ylabel('log(τ /10$^{-24}$ cm$^2$/atom)', fontsize=30)
        plt.grid(True, which='both', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), 
                  fontsize=14, 
                  loc='upper left')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        safe_print(f"Fit error: {e}")
        plt.figure(figsize=(10, 6))
        plt.scatter(Z, tau/1e-24)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Z')
        plt.ylabel('τ (10$^{-24}$ cm$^{2}$/atom)')
        plt.title('Data (Fit Failed)')
        plt.show()

    # Print final results
    safe_print("\nFinal Results:")
    safe_print("{:<5} {:<8} {:<12} {:<12}".format(
        'Element', 'Z', 'τ_mean', 'τ_std'))
    for mat, data in results.items():
        safe_print("{:<5} {:<8} {:<12.3e} {:<12.3e}".format(
            mat, data['Z'], data['tau_mean'], data['tau_std']))
else:
    safe_print("No valid results obtained")