import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
import os
from MCA_parser import MCA
import sys

sys.stdout.reconfigure(encoding='utf-8')
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
                # Extract thickness (assuming format: Material_thicknessmils_...)
                parts = file.split('_')
                t_mils = float(parts[1].replace('mils', ''))
                t_cm = t_mils * 0.00254

                # Process spectrum
                mca = MCA(os.path.join(data_dir, file))
                counts = np.array(mca.DATA)
                live_time = mca.get_live_time()
                count_rate = counts / live_time

                # Calculate ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where((N0_count_rate > 0) & (count_rate > 0),
                                    count_rate / N0_count_rate,
                                    np.nan)
                
                valid_ratio = ratio[np.isfinite(ratio)]
                if len(valid_ratio) > 0:
                    tau = calculate_tau(valid_ratio.mean(), 1, t_cm, props['rho'], props['Z'])
                    tau_values.append(tau)
                    safe_print(f"Processed {file} - τ = {tau:.3e}")
                else:
                    safe_print(f"Invalid ratio for {file}")

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
    # Prepare data for Z^4 analysis
    Z = np.array([data['Z'] for data in results.values()])
    tau = np.array([data['tau_mean'] for data in results.values()])
    tau_err = np.array([data['tau_std'] for data in results.values()])

    # Fit τ = aZ^4
    def z4_fit(Z, a, exp):
        return a * Z**exp

    try:
        popt, pcov = curve_fit(z4_fit, Z, tau, sigma=tau_err)
        a_fit = popt[0]
        Z_fit = np.linspace(min(Z), max(Z), 100)
        tau_fit = z4_fit(Z_fit, a_fit, popt[1] )

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.errorbar(np.log(Z), np.log(tau/(1e-24)), yerr=tau_err, fmt='o', label='Experimental Data')
        #plt.plot(Z_fit, tau_fit, 'r-', 
                #label=f'Fit: τ = {a_fit:.2e}Z$^{popt[1]:.2e}$')
        
        # Annotate points
        for mat, data in results.items():
            plt.annotate(mat, (data['Z'], data['tau_mean']), 
                        xytext=(5,5), textcoords='offset points')

        plt.xlabel('Atomic Number (Z)')
        plt.ylabel('Attenuation Coefficient τ (cm$^{-1}$)')
        plt.title('Photoelectric Attenuation vs Atomic Number')
        plt.yscale('log')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        safe_print(f"Error in fitting: {e}")

    # Print results
    safe_print("\nFinal Results:")
    safe_print("{:<5} {:<8} {:<12} {:<12} {:<10}".format(
        'Mat', 'Z', 'tau_mean', 'tau_std', 'N_samples'))
    for mat, data in results.items():
        safe_print("{:<5} {:<8} {:<12.3e} {:<12.3e} {:<10}".format(
            mat, data['Z'], data['tau_mean'], data['tau_std'], data['n_samples']))
else:
    safe_print("No valid results obtained from any materials")