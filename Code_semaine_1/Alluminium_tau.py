import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sys
import os
from pathlib import Path

# Ajouter le dossier parent au chemin Python
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
from MCA_parser import MCA
from scipy import constants

mca_N0 = MCA(r"semaine_1\Courant_20kV_25uA.mca")
N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time() 
N0_count_rate = N0_counts/N0_live_time
energies =  np.load("energie_semaine_1.npy")
#t_mils = np.array([10,20,30,40,50,60,70])
t_mils = np.array([1,2,3,4,5])
mils_to_cm = 0.00254 #cm/mils
t_cm = t_mils*mils_to_cm
##### TAU ######
A_Al = 13
rho_Al = 2.7
def tau(t,N,N_0= N0_counts,rho = rho_Al,A = A_Al):
    N_A = constants.N_A
    
    return (-np.ln(N/N0)/(rho*t))*(A/N_A) - 0.20*(A/N_A)
NsurN0 = []
def tau_Al(t, mu):
    return np.e**(-mu*t)

# Define threshold (e.g., 5% of max counts)
THRESHOLD_PERCENT = 0  # Adjust this value as needed

NsurN0 = []
valid_energies = []  
energy_min = 10  # keV
energy_max = 17.0  # keV

for epaisseur in t_mils:
    mca = MCA(f"semaine_1\Cu_{epaisseur}mils_20kV_25uA.mca")
    counts = np.array(mca.DATA)
    live_time = mca.get_live_time()
    dead_time = mca.get_dead_time()
    print(f"Thickness {epaisseur}mils - deadtime: {dead_time:.2f}%")
    
    count_rate = counts / live_time
    
    # Masque pour les énergies dans [7.5, 18] keV
    energy_mask = (energies >= energy_min) & (energies <= energy_max)
    
    # Appliquer le masque d'énergie ET le seuil de comptage
    combined_mask = energy_mask & (count_rate > (np.max(count_rate) * THRESHOLD_PERCENT / 100))
    
    valid_count_rate = count_rate[combined_mask]
    valid_N0_rate = N0_count_rate[combined_mask]
    valid_energy = energies[combined_mask]  # Pour vérification
    
    # Calcul du ratio N/N0 (en évitant les divisions par zéro)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((valid_N0_rate > 0) & (valid_count_rate > 0),
                        valid_count_rate / valid_N0_rate,
                        np.nan)
    
    # Stocker la plage d'énergie utilisée (pour débogage)
    if len(valid_energy) > 0:
        valid_energies.append((valid_energy.min(), valid_energy.max()))
    else:
        valid_energies.append((np.nan, np.nan))
    
    # Moyenne des ratios valides (en ignorant NaN)
    valid_ratio = ratio[np.isfinite(ratio)]
    mean_ratio = np.mean(valid_ratio) if len(valid_ratio) > 0 else np.nan
    NsurN0.append(mean_ratio)
    
# Convert to numpy arrays
NsurN0 = np.array(NsurN0)
t_cm = np.array(t_mils) * mils_to_cm
def mu(NsurN_0, t):
    return -(1/t)*np.log(NsurN_0)
print(mu(NsurN0, t_cm))
# Perform the fit
params, cov = curve_fit(tau_Al, t_cm, NsurN0)
mu_fit = params[0]
print(f"Fitted attenuation coefficient: {mu_fit:.4f} cm-1")

# Generate fit curve
fit_energie = np.linspace(0, 70*mils_to_cm, 500)
fit_tau_Al = tau_Al(fit_energie, mu_fit)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(t_cm, NsurN0, label='Experimental Data')
plt.plot(fit_energie, fit_tau_Al, 'r-', 
         label=f'Fit: μ = {mu_fit:.2f} cm⁻¹')
plt.xlabel('Aluminum Thickness (cm)')
plt.ylabel('N/N₀')
plt.title(f'X-ray Attenuation through Aluminum (Threshold: {THRESHOLD_PERCENT}% of max)')
plt.legend()
plt.grid(True)
plt.show()

# Print energy ranges used
print("\nEnergy ranges used for each thickness:")
for i, (e_min, e_max) in enumerate(valid_energies):
    print(f"{t_mils[i]} mils: {e_min:.2f}-{e_max:.2f} keV")

