import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from MCA_parser import MCA
from scipy import constants
from scipy.signal import savgol_filter

# Load NIST data (replace with your actual NIST data loading)
# Expected format: 2 columns [energy_keV, mu_cm2perg]
nist_data = np.loadtxt('NIST_AL_coeff_total.csv', skiprows=2)  # Skip 2 header lines
nist_energy_MeV = nist_data[:, 0]  # 1st column: Energy in MeV
nist_mu = nist_data[:, 1]          # 2nd column: μ/ρ in cm²/g

# Convert MeV to keV and filter to 0.5-20 keV range
nist_energy_keV = nist_energy_MeV * 1000  
valid_mask = (nist_energy_keV >= 0.5) & (nist_energy_keV <= 20)
nist_energy_keV = nist_energy_keV[valid_mask]
nist_mu = nist_mu[valid_mask]

# Verify array lengths match
assert len(nist_energy_keV) == len(nist_mu), "NIST data dimension mismatch"

# Convert to linear attenuation coefficient (cm⁻¹)
rho_Al = 2.7  # g/cm³ density of aluminum
nist_tau = nist_mu * rho_Al

# Load your experimental data
mca_N0 = MCA(r"semaine_1\Courant_20kV_25uA.mca")
N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time() 
N0_count_rate = N0_counts / N0_live_time
energies = np.load("energie_semaine_1.npy")

# Filter your experimental data to 0.5-20 keV
exp_mask = (energies >= 1.5) & (energies <=18)
valid_energies = energies[exp_mask]
valid_N0_rate = N0_count_rate[exp_mask]

# Thickness parameters
t_mils = np.array([10, 20, 30, 40, 50, 60, 70])
mils_to_cm = 0.00254
t_cm = t_mils * mils_to_cm

# Process experimental data
n_energies = len(valid_energies)
tau_exp = np.full(n_energies, np.nan)
n_valid = np.zeros(n_energies)

for i, energy in enumerate(valid_energies):
    ratios = []
    valid_t = []
    
    for thickness_cm in t_cm:
        epaisseur = int(round(thickness_cm / mils_to_cm))
        mca = MCA(f"semaine_1\Al_{epaisseur}mils_20kV_25uA.mca")
        counts = np.array(mca.DATA)[exp_mask]
        live_time = mca.get_live_time()
        count_rate = counts / live_time
        
        if (valid_N0_rate[i] > 0) and (count_rate[i] > 0):
            ratios.append(count_rate[i] / valid_N0_rate[i])
            valid_t.append(thickness_cm)
    
    if len(ratios) >= 3:
        try:
            params, _ = curve_fit(lambda t, mu: np.exp(-mu * t), valid_t, ratios)
            tau_exp[i] = params[0]
            n_valid[i] = len(ratios)
        except RuntimeError:
            continue

# Smooth experimental results
window_size = min(51, len(tau_exp[~np.isnan(tau_exp)])//2*2+1)
if window_size > 3:
    tau_smoothed = savgol_filter(tau_exp[~np.isnan(tau_exp)], window_size, 2)
else:
    tau_smoothed = tau_exp[~np.isnan(tau_exp)]

# Create plot
plt.figure(figsize=(12, 6))

# Plot NIST reference
plt.plot(nist_energy_keV, nist_tau, 'k-', lw=2, label='NIST Reference')

# Plot experimental data
valid_points = ~np.isnan(tau_exp)
plt.plot(valid_energies[valid_points], tau_exp[valid_points], 
         'bo', ms=4, alpha=0.5, label='Experimental Points')
#plt.plot(valid_energies[valid_points], tau_smoothed, 
         #'r-', lw=2, label='Experimental Smoothed')

# Highlight Aluminum K-edge at 1.56 keV
plt.axvline(1.56, color='purple', linestyle=':', alpha=0.7, label='Al K-edge (1.56 keV)')
#plt.scatter(valid_energies[valid_points], tau_exp[valid_points] )
plt.xlabel('Energy (keV)')
plt.ylabel('Attenuation Coefficient τ (cm⁻¹)')
plt.title('Aluminum Attenuation Coefficient (0.5-20 keV)')
plt.yscale('log')
plt.grid(True, which='both', alpha=0.3)
plt.legend()



plt.tight_layout()
plt.show()